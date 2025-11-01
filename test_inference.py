"""
Quick test harness for backend/inference.py

Usage (from repo root, preferably inside your .venv):

# check text prediction
python backend/test_inference.py --text "heavy flooding near river bank"

# check image prediction (requires torch & torchvision & pillow and a model checkpoint or will fallback)
python backend/test_inference.py --image path/to/sample.jpg

Note: this script does not install torch for you. If you want CPU-only PyTorch on Windows, visit https://pytorch.org/get-started/locally/ and pick the appropriate command for "Stable" + "Windows" + "Pip" + "CPU".
"""
import argparse
import pprint

from backend import inference

pp = pprint.PrettyPrinter(indent=2)

parser = argparse.ArgumentParser(description='Test inference functions')
parser.add_argument('--text', type=str, help='Text to classify')
parser.add_argument('--image', type=str, help='Path to image to classify')
args = parser.parse_args()

if args.text:
    print('\n== predict_text result ==')
    res = inference.predict_text(args.text)
    pp.pprint(res)

if args.image:
    print('\n== predict_image result ==')
    res = inference.predict_image(args.image)
    pp.pprint(res)

if not (args.text or args.image):
    print('Provide --text or --image to test. Example:')
    print('  python backend/test_inference.py --text "smoke and flames in a market"')
