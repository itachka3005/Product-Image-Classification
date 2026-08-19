import torch
from pathlib import Path
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset_prepared" / "test"
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
CONFUSION_MATRIX_PATH = BASE_DIR / "output" / "confusion_matrix.png"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate():
    if not MODEL_PATH.exists():
        print(f"Ошибка: Файл модели {MODEL_PATH} не найден. Сначала запустите train.py!")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()


    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_preds = []
    all_labels = []

    print("Запуск тестирования...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

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