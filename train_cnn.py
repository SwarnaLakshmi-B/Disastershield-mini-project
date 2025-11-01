"""
Fine-tune ResNet50 on a directory of images arranged as expected by torchvision.datasets.ImageFolder:

  data/
    train/
      flood/
      fire/
      no_event/
    val/
      flood/
      fire/
      no_event/

Saves the best checkpoint to `--output` (default: backend/models/best_cnn.pth`).

This script is intentionally conservative: it will print actionable errors if
PyTorch/torchvision are not installed rather than raising import errors.
"""
import argparse
import os
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
except Exception:
    torch = None


def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 on an ImageFolder dataset')
    parser.add_argument('--data-dir', required=True, help='Path to data directory containing train/ and val/ folders')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', default=os.path.join('backend', 'models', 'best_cnn.pth'))
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    if torch is None:
        print('PyTorch/torchvision not available. Install with: pip install torch torchvision')
        return

    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print('Expected train/ and val/ directories under --data-dir')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(train_ds.classes)
    print('Detected classes:', train_ds.classes)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_acc = 0.0
    ensure_dir(args.output)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total if total else 0.0
        elapsed = time.time() - start
        print(f'Epoch {epoch}/{args.epochs} loss={epoch_loss:.4f} val_acc={acc:.4f} time={elapsed:.1f}s')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.output)
            print('Saved best model to', args.output)

    print('Training complete. Best val acc:', best_acc)


if __name__ == '__main__':
    main()
