"""
Fine-tuning google/vit-base-patch16-224-in21k для классификации товаров.

Адаптировано под структуру проекта (dataset_prepared/train|val|test/<class>/*.jpg)
по мотивам https://github.com/pouyaSamie/fine-tune-google-vit-base-patch16-224-in21k
(тот репозиторий, в свою очередь, следует официальному туториалу Hugging Face
по fine-tuning ViT: https://huggingface.co/blog/fine-tune-vit).

Главные отличия от прежнего train.py на ResNet18:
- вместо torchvision.models.resnet18 используется предобученный
  google/vit-base-patch16-224-in21k из HF Hub
- обучение идёт через transformers.Trainer вместо ручного цикла
- модель сохраняется в формате Hugging Face (папка с config.json,
  model.safetensors, preprocessor_config.json) через save_pretrained,
  а не как единый .pth-чекпоинт
"""

from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torchvision.transforms import (
    ColorJitter, Compose, Normalize, RandomHorizontalFlip,
    RandomResizedCrop, RandomRotation, Resize, ToTensor,
)
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

BASE_DIR = Path(__file__).resolve().parent
PREPARED_DIR = BASE_DIR / "dataset_prepared"
MODEL_SAVE_DIR = BASE_DIR / "models" / "vit-product-classifier"

MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 16          # ViT тяжелее resnet18 — батч поменьше, особенно на CPU
NUM_EPOCHS = 8
LEARNING_RATE = 2e-5     # для fine-tune предобученного трансформера нужен маленький LR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    data_files = {
        "train": str(PREPARED_DIR / "train" / "*" / "*"),
        "validation": str(PREPARED_DIR / "val" / "*" / "*"),
        "test": str(PREPARED_DIR / "test" / "*" / "*"),
    }
    return load_dataset("imagefolder", data_files=data_files)


def build_transforms(processor: ViTImageProcessor):
    mean, std = processor.image_mean, processor.image_std
    size = processor.size.get("height", processor.size.get("shortest_edge", 224))

    train_tf = Compose([
        RandomResizedCrop(size, scale=(0.8, 1.0)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    eval_tf = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


def make_transform_fn(torchvision_tf):
    def _transform(example_batch):
        example_batch["pixel_values"] = [
            torchvision_tf(img.convert("RGB")) for img in example_batch["image"]
        ]
        example_batch["labels"] = example_batch["label"]
        return example_batch
    return _transform


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)


def main():
    print(f"Устройство: {DEVICE}")

    dataset = load_data()
    labels = dataset["train"].features["label"].names
    print(f"Классы ({len(labels)}): {labels}")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    train_tf, eval_tf = build_transforms(processor)

    dataset["train"] = dataset["train"].with_transform(make_transform_fn(train_tf))
    dataset["validation"] = dataset["validation"].with_transform(make_transform_fn(eval_tf))
    dataset["test"] = dataset["test"].with_transform(make_transform_fn(eval_tf))

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={str(i): name for i, name in enumerate(labels)},
        label2id={name: str(i) for i, name in enumerate(labels)},
    ).to(DEVICE)

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
    )

    trainer.train()

    print("\n=== Оценка на test ===")
    test_metrics = trainer.evaluate(dataset["test"])
    print(test_metrics)

    trainer.save_model(str(MODEL_SAVE_DIR))
    processor.save_pretrained(str(MODEL_SAVE_DIR))
    print(f"\nМодель и процессор сохранены в: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()