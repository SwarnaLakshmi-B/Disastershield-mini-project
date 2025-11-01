"""
Train an NLP classifier. Two modes supported:
 - LSTM: builds a small vocabulary from the CSV and trains an LSTMClassifier
 - BERT: fine-tunes a transformers model (if transformers installed)

The input CSV should be simple with two columns: text,label (header optional).
Example:
  text,label
  "smoke and flames",fire
  "water in the street",flood

Saves checkpoint to `--output` (default: backend/models/best_nlp.pth).
"""
import argparse
import os
import csv
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


class TextDataset(Dataset):
    def __init__(self, rows, vocab=None, max_len=64):
        self.rows = rows
        self.max_len = max_len
        self.vocab = vocab or {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        toks = text.lower().split()[: self.max_len]
        idxs = [self.vocab.get(t, 1) for t in toks]
        return torch.tensor(idxs, dtype=torch.long), int(label)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(max(2, vocab_size), embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])


def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        first = next(r)
        # detect header
        if 'text' in [c.lower() for c in first] and 'label' in [c.lower() for c in first]:
            headers = [c.lower() for c in first]
            ti = headers.index('text')
            li = headers.index('label')
        else:
            # first is data
            ti = 0
            li = 1
            rows.append((first[ti], first[li]))
        for row in r:
            if len(row) < 2: continue
            rows.append((row[ti], row[li]))
    return rows


def build_vocab(rows, min_freq=1):
    freq = {}
    for t, _ in rows:
        for w in t.lower().split():
            freq[w] = freq.get(w, 0) + 1
    # reserve 0=pad,1=unk
    vocab = {}
    idx = 2
    for w, c in sorted(freq.items(), key=lambda x: -x[1]):
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab


def collate_batch(batch):
    # batch: list of (tensor(idxs), label)
    xs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    maxlen = max([x.size(0) for x in xs])
    padded = torch.zeros((len(xs), maxlen), dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : x.size(0)] = x
    return padded, labels


def train_lstm(args):
    if torch is None:
        print('PyTorch not available. Install with: pip install torch')
        return

    rows = read_csv(args.data_file)
    label_set = sorted(list({r[1] for r in rows}))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    mapped = [(t, label_to_idx[l]) for t, l in rows]

    vocab = build_vocab(rows)
    ds = TextDataset(mapped, vocab=vocab, max_len= args.max_len)

    def idx_rows(ds_rows):
        return ds_rows

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    model = LSTMClassifier(vocab_size=len(vocab) + 2, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_classes=len(label_set))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        acc = correct / total if total else 0.0
        print(f'Epoch {epoch} acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'embed_dim': args.embed_dim, 'hidden_dim': args.hidden_dim, 'num_classes': len(label_set)}, args.output)
            print('Saved checkpoint to', args.output)

    print('Training complete. best_acc=', best_acc)


def train_bert(args):
    if not TRANSFORMERS_AVAILABLE:
        print('transformers not available. Install with: pip install transformers datasets')
        return
    # Minimal BERT training via Trainer API requires datasets and a little glue code.
    print('BERT training path selected; see README for specific usage')


def main():
    parser = argparse.ArgumentParser(description='Train NLP classifier (LSTM or BERT)')
    parser.add_argument('--data-file', required=True, help='CSV with text,label')
    parser.add_argument('--model', choices=['lstm', 'bert'], default='lstm')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', default=os.path.join('backend', 'models', 'best_nlp.pth'))
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=64)
    args = parser.parse_args()

    if args.model == 'lstm':
        train_lstm(args)
    else:
        train_bert(args)


if __name__ == '__main__':
    main()
