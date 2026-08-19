"""
Оценка дообученного ViT на test-сплите.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

BASE_DIR = Path(__file__).resolve().parent
PREPARED_DIR = BASE_DIR / "dataset_prepared"
MODEL_DIR = BASE_DIR / "models" / "vit-product-classifier"
CONFUSION_MATRIX_PATH = BASE_DIR / "output" / "confusion_matrix.png"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch, processor):
    images = [x["image"].convert("RGB") for x in batch]
    labels = torch.tensor([x["label"] for x in batch])
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    return pixel_values, labels


def evaluate():
    if not MODEL_DIR.exists():
        print(f"Ошибка: папка модели {MODEL_DIR} не найдена. Сначала запусти train.py!")
        return

    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE).eval()
    class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    test_dataset = load_dataset(
        "imagefolder", data_files={"test": str(PREPARED_DIR / "test" / "*" / "*")}
    )["test"]

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    all_preds = []
    all_labels = []

    print("Запуск тестирования...")
    with torch.no_grad():
        for pixel_values, labels in test_loader:
            pixel_values = pixel_values.to(DEVICE)
            outputs = model(pixel_values=pixel_values)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 60)
    print("               CLASSIFICATION REPORT (TEST)")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    cm = confusion_matrix(all_labels, all_preds)

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("Истинный класс (True)")
    plt.xlabel("Предсказанный класс (Predicted)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)

    print(f"Матрица ошибок сохранена в: {CONFUSION_MATRIX_PATH}\n")


if __name__ == "__main__":
    evaluate()