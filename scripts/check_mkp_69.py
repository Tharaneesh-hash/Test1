"""
scripts/check_mkp_69.py — Compare predictions on MKP_69 images.
"""
import sys
sys.path.insert(0, ".")
import torch
from PIL import Image
from torchvision import transforms
from src.config import BEST_BINARY_MODEL_PATH
from src.model import load_binary_model

device = torch.device("cpu")
model = load_binary_model(BEST_BINARY_MODEL_PATH, device=device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for fname in ["MKP_69_01.jpg", "MKP_69_02.jpg", "MKP_69_03.jpg"]:
    p = f"FOLDS/fold1/test/{fname}"
    im = Image.open(p).convert("RGB")
    t = tf(im).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_proba(t)[0].numpy()
    print(f"{fname}: Mpox Prob = {probs[1]:.2%}, Non-Mpox Prob = {probs[0]:.2%}")
