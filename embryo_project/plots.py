from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from embryo_project.config import FIGURES_DIR, REPORTS_DIR, PROCESSED_DATA_DIR
import typer

app = typer.Typer()

# ---------- HELPERS ----------

def folder_contains_class1(folder: Path) -> bool:
    """Check if a folder contains at least one _1.jpg image."""
    return any(img.stem.endswith("_1") for img in folder.glob("*.jpg"))

def scan_split(split_dir: Path, include_aug: bool = False):
    """Scan one split (train/val/test) and count folders and images by class."""
    class0_folders = class1_folders = 0
    class0_images = class1_images = 0

    for folder in split_dir.iterdir():
        if folder.is_dir():
            # Skip augmented folders if include_aug is False
            if not include_aug and "_aug" in folder.name:
                continue

            # Folder-level class detection
            if folder_contains_class1(folder):
                class1_folders += 1
            else:
                class0_folders += 1

            # Image-level counts
            for img_path in folder.glob("*.jpg"):
                if img_path.stem.endswith("_1"):
                    class1_images += 1
                else:
                    class0_images += 1

    return {
        "folders": {"class0": class0_folders, "class1": class1_folders},
        "images": {"class0": class0_images, "class1": class1_images},
    }

def scan_dataset(data_dir: Path, include_aug: bool = False):
    """Scan the whole dataset (train, val, test, train_balanced)."""
    stats = {}
    for split in data_dir.iterdir():
        if split.is_dir():
            stats[split.name] = scan_split(split, include_aug=include_aug)
    return stats

# ---------- PLOTTING ----------

def plot_data_distribution(data_dir: Path = PROCESSED_DATA_DIR,
                           save_path: Path = None,
                           include_aug: bool = False):
    stats = scan_dataset(data_dir, include_aug=include_aug)
    logger.info(f"Scanned {data_dir}")

    splits = list(stats.keys())
    width = 0.15  # width of each bar
    x = [0, 1]   # Class 0 and Class 1

    # Prepare values: for each split, a list [class0_count, class1_count]
    folder_values = [[stats[s]["folders"]["class0"], stats[s]["folders"]["class1"]] for s in splits]
    image_values  = [[stats[s]["images"]["class0"], stats[s]["images"]["class1"]] for s in splits]

    # --- Folders Distribution (classes on x) ---
    plt.figure(figsize=(10, 6))
    for i, (split, vals) in enumerate(zip(splits, folder_values)):
        plt.bar([xi + (i - len(splits)/2)*width + width/2 for xi in x], vals,
                width, label=split)

    plt.xticks(x, ["Class 0", "Class 1"])
    plt.ylabel("Number of Folders")
    plt.title("Folder Distribution per Split")
    plt.legend()

    # Annotate counts
    for i, vals in enumerate(folder_values):
        for j, val in enumerate(vals):
            xpos = x[j] + (i - len(splits)/2)*width + width/2
            plt.text(xpos, val, str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "folders_distribution_x_classes.png", bbox_inches="tight")
        logger.success("Saved folders_distribution_x_classes.png")
    else:
        plt.show()
    plt.close()

    # --- Images Distribution (classes on x) ---
    plt.figure(figsize=(10, 6))
    for i, (split, vals) in enumerate(zip(splits, image_values)):
        plt.bar([xi + (i - len(splits)/2)*width + width/2 for xi in x], vals,
                width, label=split)  # no color specified

    plt.xticks(x, ["Class 0", "Class 1"])
    plt.ylabel("Number of Images")
    plt.title("Image Distribution per Split")
    plt.legend()

    # Annotate counts
    for i, vals in enumerate(image_values):
        for j, val in enumerate(vals):
            xpos = x[j] + (i - len(splits)/2)*width + width/2
            plt.text(xpos, val, str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    if save_path:
        plt.savefig(save_path / "images_distribution_x_classes.png", bbox_inches="tight")
        logger.success("Saved images_distribution_x_classes.png")
    else:
        plt.show()
    plt.close()

    # --- Folders Distribution (splits on x) ---
    class0_folders = [stats[s]["folders"]["class0"] for s in splits]
    class1_folders = [stats[s]["folders"]["class1"] for s in splits]

    x = range(len(splits))
    width = 0.35

    plt.figure(figsize=(8, 6))
    bars0 = plt.bar([i - width/2 for i in x], class0_folders, width, label="Class 0")
    bars1 = plt.bar([i + width/2 for i in x], class1_folders, width, label="Class 1")

    plt.xticks(x, splits)
    plt.ylabel("Number of Folders")
    plt.title("Folder Distribution per Split")
    plt.legend()

    # Annotate counts
    for bars in [bars0, bars1]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     str(height), ha="center", va="bottom", fontsize=9, fontweight="bold")

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "folders_distribution_x_splits.png")
        logger.success(f"Saved folders_distribution_x_splits.png")
    else:
        plt.show()
    plt.close()

    # --- Images Distribution (splits on x) ---
    class0_images = [stats[s]["images"]["class0"] for s in splits]
    class1_images = [stats[s]["images"]["class1"] for s in splits]

    plt.figure(figsize=(8, 6))
    bars0 = plt.bar([i - width/2 for i in x], class0_images, width, label="Class 0")
    bars1 = plt.bar([i + width/2 for i in x], class1_images, width, label="Class 1")

    plt.xticks(x, splits)
    plt.ylabel("Number of Images")
    plt.title("Image Distribution per Split")
    plt.legend()

    # Annotate counts
    for bars in [bars0, bars1]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     str(height), ha="center", va="bottom", fontsize=9, fontweight="bold")

    if save_path:
        plt.savefig(save_path / "images_distribution_x_splits.png")
        logger.success(f"Saved images_distribution_x_splits.png")
    else:
        plt.show()
    plt.close()


def plot_classification_results(preds, labels, model_name, figures_dir=FIGURES_DIR):
    # Classification report
    report = classification_report(labels, preds, target_names=["Class 0", "Class 1"], digits=4)
    report_path = REPORTS_DIR / f"{model_name}_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.success(f"Classification report saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    cm_path = figures_dir / f"{model_name}_cm.png"
    plt.savefig(cm_path)
    plt.close()
    logger.success(f"Confusion matrix saved to {cm_path}")


# ---------- CLI ----------

@app.command()
def plot_distribution(
    data_dir: Path = PROCESSED_DATA_DIR,
    save: bool = typer.Option(False, help="If set, save plots instead of showing"),
    output_dir: Path = FIGURES_DIR,
    include_aug: bool = typer.Option(True, help="Include augmented folders in counts")
):
    """Plot data distribution (folders + images) for train/val/test splits."""
    plot_data_distribution(data_dir,
                           save_path=output_dir if save else None,
                           include_aug=include_aug)

if __name__ == "__main__":
    app()
