from pathlib import Path
from loguru import logger
import shutil
import random
from PIL import Image
import typer
import torchvision.transforms as T

app = typer.Typer()

# ---------- AUGMENTATION TRANSFORMS ----------

augmentations = [
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
]

# ---------- SUBFUNCTIONS ----------

def folder_contains_class1(folder: Path) -> bool:
    """Check if a folder contains at least one image ending with _1.jpg"""
    for img_path in folder.glob("*.jpg"):
        if img_path.stem.endswith("_1"):  # stem = filename without extension
            return True
    return False

def augment_folder(folder_path: Path, num_aug: int = 2):
    """Create augmented copies of all images in one folder."""
    images = list(folder_path.glob("*.jpg"))
    if not images:
        logger.warning(f"No JPG images found in {folder_path}")
        return

    for aug_idx in range(1, num_aug + 1):
        aug_transform = random.choice(augmentations)
        new_folder = folder_path.parent / f"{folder_path.name}_aug{aug_idx}"
        new_folder.mkdir(exist_ok=True)

        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            aug_img = aug_transform(img)
            aug_img.save(new_folder / img_path.name, "JPEG")

        logger.success(f"Created {new_folder}")

def augment_dataset(train_dir: Path, num_aug: int = 2, only_class1: bool = False):
    """Augment dataset: all folders or only those that contain _1 images."""
    for folder in train_dir.iterdir():
        if folder.is_dir() and "_aug" not in folder.name:
            if only_class1:
                if folder_contains_class1(folder):
                    logger.info(f"Augmenting class-1 folder: {folder.name}")
                    augment_folder(folder, num_aug=num_aug)
                else:
                    logger.debug(f"Skipping {folder.name} (no _1 images found)")
            else:
                augment_folder(folder, num_aug=num_aug)

def remove_augmented_folders(train_dir: Path):
    """Remove all augmented folders (ending with '_augX')."""
    removed = 0
    for folder in train_dir.iterdir():
        if folder.is_dir() and "_aug" in folder.name:
            shutil.rmtree(folder)
            logger.info(f"Removed {folder}")
            removed += 1
    logger.success(f"Removed {removed} augmented folders from {train_dir}")


# ---------- TYPER COMMANDS ----------

@app.command()
def augmentation(
    train_dir: Path,
    num_aug: int = 2,
    only_class1: bool = typer.Option(False, help="If True, augment only folders containing _1 images")
):
    """Augment train dataset by creating new folders with transformed images."""
    augment_dataset(train_dir, num_aug=num_aug, only_class1=only_class1)

@app.command()
def clean(
    train_dir: Path
):
    """Remove all augmented folders from train dataset."""
    remove_augmented_folders(train_dir)


if __name__ == "__main__":
    app()
