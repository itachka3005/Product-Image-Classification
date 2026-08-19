import hashlib
import random
import shutil
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
RAW_DATASET_DIR = BASE_DIR / "dataset"
PREPARED_DIR = BASE_DIR / "dataset_prepared"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False

def get_image_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def prepare_dataset():
    random.seed(SEED)
    global_seen_hashes = set()

    for category_dir in RAW_DATASET_DIR.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        valid_images = []

        for img_path in category_dir.glob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            if not is_valid_image(img_path):
                continue

            img_hash = get_image_hash(img_path)
            if img_hash in global_seen_hashes:
                continue

            global_seen_hashes.add(img_hash)
            valid_images.append(img_path)

        random.shuffle(valid_images)
        n_total = len(valid_images)
        n_train = int(n_total * SPLIT_RATIOS["train"])
        n_val = int(n_total * SPLIT_RATIOS["val"])

        splits = {
            "train": valid_images[:n_train],
            "val": valid_images[n_train:n_train + n_val],
            "test": valid_images[n_train + n_val:]
        }

        for split_name, files in splits.items():
            dest_dir = PREPARED_DIR / split_name / category_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file_path in files:
                shutil.copy2(file_path, dest_dir / file_path.name)

        print(f"[{category_name:<12}] Валидных: {n_total:>3} | Train: {len(splits['train']):>3} | Val: {len(splits['val']):>3} | Test: {len(splits['test']):>3}")

if __name__ == "__main__":
    prepare_dataset()
