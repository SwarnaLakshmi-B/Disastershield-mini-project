# model inference logic
# backend/inference.py
"""
Simple inference module.
- load_models() returns a dict of model placeholders (so signature won't change when you add real models).
- predict_text(text, location, models=...) returns a dictionary:
    { label, prob, severity, source, original_text, location }
Replace internals with a real transformer / DL inference later.
"""

from typing import Optional, Dict
import os
import json
import math

# Try to import torch/vision; if unavailable we'll gracefully fall back to the
# existing lightweight keyword classifier so functions still run without error.
try:
    import torch
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Optional transformers (zero-shot) support for text classification
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# keyword-based mock classifier (fallback)
KEYWORDS = {
    "flood": ["flood", "flooded", "water", "river overflow", "submerged", "overflow"],
    "fire": ["fire", "smoke", "burning", "flames", "wildfire"],
    "earthquake": ["earthquake", "tremor", "shake", "collapsed", "quake"],
    "landslide": ["landslide", "slide", "mudslide", "mass movement"]
}

# canonical labels used by the project
LABELS = ["flood", "fire", "no_event"]


def load_models():
    """Return dict of loaded models if available.

    This will attempt to lazily load two artifacts if present:
      - outputs/best_cnn.pth  (or backend/models/best_cnn.pth)
      - outputs/best_nlp.pth  (or backend/models/best_nlp.pth)

    If torch is unavailable or files are missing we return an empty dict.
    """
    models_out = {}
    if not TORCH_AVAILABLE:
        return models_out

    # CNN: ResNet50 fine-tuned to len(LABELS)
    cnn_paths = [os.path.join("outputs", "best_cnn.pth"), os.path.join("backend", "models", "best_cnn.pth")]
    for p in cnn_paths:
        if os.path.exists(p):
            try:
                net = models.resnet50(weights=None)
                # replace final layer to match our label set
                import torch.nn as nn
                net.fc = nn.Linear(net.fc.in_features, len(LABELS))
                net.load_state_dict(torch.load(p, map_location="cpu"))
                net.eval()
                models_out["cnn"] = net
                models_out["cnn_path"] = p
                break
            except Exception as e:
                print("[inference] failed to load cnn from", p, e)

    # NLP: simple LSTM classifier saved as best_nlp.pth
    nlp_paths = [os.path.join("outputs", "best_nlp.pth"), os.path.join("backend", "models", "best_nlp.pth")]
    for p in nlp_paths:
        if os.path.exists(p):
            try:
                # consumer code will call predict_text which will import/construct the
                # model. We just expose the path here for later lazy-loading.
                models_out["nlp_path"] = p
                break
            except Exception as e:
                print("[inference] failed to register nlp model", p, e)

    return models_out


def health_check() -> bool:
    # If you load real models, perform a lightweight sanity check here.
    return True


def predict_image(image_path: str):
    """Predict an image using a ResNet50-based classifier.

    Behavior:
      - if torch is not available, returns a safe no-event prediction
      - if a fine-tuned checkpoint exists (best_cnn.pth), loads it with
        a final linear layer sized to len(LABELS) and predicts.
      - otherwise returns a harmless no_event with prob 0.0 so caller still
        receives a valid JSON-friendly structure.
    """
    if not TORCH_AVAILABLE:
        return {"label": "no_event", "prob": 0.0, "source": "image", "note": "torch not available"}

    # prefer backend/models then outputs
    cnn_paths = [os.path.join("backend", "models", "best_cnn.pth"), os.path.join("outputs", "best_cnn.pth")]
    ckpt = None
    for p in cnn_paths:
        if os.path.exists(p):
            ckpt = p
            break

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {"label": "no_event", "prob": 0.0, "source": "image", "note": f"failed to open image: {e}"}

    x = preprocess(img).unsqueeze(0)

    if ckpt:
        try:
            net = models.resnet50(weights=None)
            import torch.nn as nn
            net.fc = nn.Linear(net.fc.in_features, len(LABELS))
            net.load_state_dict(torch.load(ckpt, map_location="cpu"))
            net.eval()
            with torch.no_grad():
                logits = net(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                prob_val, idx = torch.max(probs, dim=0)
                label = LABELS[int(idx)] if int(idx) < len(LABELS) else "no_event"
                return {"label": label, "prob": float(prob_val.item()), "source": "image", "model": ckpt}
        except Exception as e:
            return {"label": "no_event", "prob": 0.0, "source": "image", "note": f"failed to run cnn: {e}"}

    # no checkpoint: fallback
    return {"label": "no_event", "prob": 0.0, "source": "image", "note": "no fine-tuned cnn available"}


# Simple LSTM classifier definition (if you prefer to save and load weights)
if TORCH_AVAILABLE:
    import torch.nn as nn

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


def predict_text(text: str, location: Optional[str] = None, models: Optional[Dict] = None):
    """Text prediction. Attempts to use a saved LSTM model if present; otherwise
    falls back to the lightweight keyword-based classifier.

    Returns a dict: { label, prob, severity, source, original_text, location }
    """
    # Try transformers zero-shot classification first (if available).
    if TRANSFORMERS_AVAILABLE:
        try:
            classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
            out = classifier((text or ""), candidate_labels=LABELS)
            # out contains 'labels' and 'scores'
            lbls = out.get('labels') or []
            scores = out.get('scores') or []
            if lbls and scores:
                label = lbls[0]
                prob_val = float(scores[0])
                severity = 'high' if prob_val >= 0.8 else 'medium' if prob_val >= 0.55 else 'low'
                return {"label": label, "prob": prob_val, "severity": severity, "source": "transformers_zero_shot", "original_text": text, "location": location}
        except Exception as e:
            # fallback to other methods
            print('[inference] transformers zero-shot failed', e)

    # If torch and a saved LSTM model exists, try to load and run it. Otherwise use
    # the original keyword heuristic so this function never raises an error.
    if TORCH_AVAILABLE:
        # try to find a saved NLP model first
        nlp_paths = [os.path.join("backend", "models", "best_nlp.pth"), os.path.join("outputs", "best_nlp.pth")]
        for p in nlp_paths:
            if os.path.exists(p):
                try:
                    # For a dropped-in example we assume the saved checkpoint contains:
                    # { 'state_dict': ..., 'vocab': {token:idx}, 'embed_dim':int, 'hidden_dim':int }
                    ckpt = torch.load(p, map_location="cpu")
                    vocab = ckpt.get('vocab') or {}
                    state = ckpt.get('state_dict') or ckpt
                    embed_dim = ckpt.get('embed_dim', 64)
                    hidden_dim = ckpt.get('hidden_dim', 64)
                    num_classes = ckpt.get('num_classes', len(LABELS))

                    # build model and load weights
                    model = LSTMClassifier(max(2, len(vocab)), embed_dim, hidden_dim, num_classes)
                    model.load_state_dict(state)
                    model.eval()

                    # simple whitespace tokenizer using the saved vocab
                    toks = (text or "").lower().split()
                    idxs = [vocab.get(t, 1) for t in toks]  # 1 = unk
                    if not idxs:
                        return {"label": "no_event", "prob": 0.0, "severity": "low", "source": "text_lstm", "original_text": text, "location": location}

                    import torch.nn.functional as F
                    x = torch.tensor([idxs], dtype=torch.long)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1).squeeze(0)
                        prob_val, idx = torch.max(probs, dim=0)
                        label = LABELS[int(idx)] if int(idx) < len(LABELS) else "no_event"
                        prob_val = float(prob_val.item())
                        severity = 'high' if prob_val >= 0.8 else 'medium' if prob_val >= 0.55 else 'low'
                        return {"label": label, "prob": prob_val, "severity": severity, "source": "text_lstm", "original_text": text, "location": location}
                except Exception as e:
                    # fallthrough to keyword heuristic
                    print("[inference] failed to run nlp model", p, e)
                    break

    # fallback keyword heuristic (original behavior)
    t = (text or "").lower()
    votes = {}
    for label, kws in KEYWORDS.items():
        for kw in kws:
            if kw in t:
                votes[label] = votes.get(label, 0) + 1

    if not votes:
        label = "no_event"
        prob = 0.30
        severity = "low"
    else:
        label = max(votes, key=lambda k: votes[k])
        # pseudo probability: more keyword hits -> higher confidence
        base = 0.55
        count = votes[label]
        prob = min(base + 0.18 * (count - 1) + 0.05 * count, 0.99)
        if prob > 0.80:
            severity = "high"
        elif prob > 0.55:
            severity = "medium"
        else:
            severity = "low"

    return {
        "label": label,
        "prob": round(prob, 2),
        "severity": severity,
        "source": "text",
        "original_text": text,
        "location": location
    }


