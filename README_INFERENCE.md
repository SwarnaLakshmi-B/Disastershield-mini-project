Inference module notes

Files
- backend/inference.py  -- contains predict_text() and predict_image(), plus utilities to load checkpoints if present.
- backend/test_inference.py -- small CLI to run quick checks.

Quick checks

1) Activate your project's virtualenv (.venv):

Windows PowerShell:

```
.\.venv\Scripts\Activate.ps1
```

2) (Optional) Install dependencies for image/text models if you plan to use them (PyTorch + torchvision + Pillow). Pick the official command from https://pytorch.org/get-started/locally/ for the correct platform and CUDA/CPU choices. Example (CPU-only, may change):

```
pip install torch torchvision pillow
```

3) Run the test script:

```
python backend/test_inference.py --text "heavy flooding near river bank"
python backend/test_inference.py --image path/to/sample.jpg
```

Notes
- The code in `inference.py` is defensive: if PyTorch/torchvision/PIL are not installed the functions fall back to the original keyword heuristic for text and a safe `no_event` for images.
- To enable real image predictions, place a fine-tuned ResNet checkpoint at either `backend/models/best_cnn.pth` or `outputs/best_cnn.pth`.
-- To enable LSTM-based text classification, place a compatible checkpoint at `backend/models/best_nlp.pth` or `outputs/best_nlp.pth` (see code comments for expected keys like `vocab`, `state_dict`, `embed_dim`, `hidden_dim`).

Training scripts

Two helper training scripts are provided under `backend/`:

- `backend/train_cnn.py` — fine-tune a ResNet50 on an ImageFolder layout (train/ and val/). Example:

```powershell
python backend/train_cnn.py --data-dir path\to\data --epochs 10 --batch-size 32 --output backend\models\best_cnn.pth
```

- `backend/train_nlp.py` — train an LSTM text classifier from a CSV of text,label pairs or (optionally) fine-tune a transformer if `transformers` is installed. Example (LSTM):

```powershell
python backend/train_nlp.py --data-file data\labels.csv --model lstm --epochs 10 --output backend\models\best_nlp.pth
```

Install ML deps

See `backend/requirements-ml.txt` for the optional packages used by the training scripts. Installing PyTorch often requires selecting the wheel for your OS and CUDA capability — visit https://pytorch.org/get-started/locally/ and follow the recommended install command.
