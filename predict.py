"""
Инференс через дообученный google/vit-base-patch16-224-in21k.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "vit-product-classifier"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(img_path: str):
    if not MODEL_DIR.exists():
        print(f"Ошибка: папка модели {MODEL_DIR} не найдена. Сначала запусти train.py!")
        return

    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE).eval()

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        top_prob, top_class_idx = torch.max(probs, 0)

    class_name = model.config.id2label[top_class_idx.item()]
    print(f"\nИзображение: {img_path}")
    print(f"Предсказание: {class_name} ({top_prob.item() * 100:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Использование: python predict.py <путь_к_картинке>")