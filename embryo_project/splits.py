from pathlib import Path
from loguru import logger
import typer
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from embryo_project.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, ANNOTATIONS_FILE

app = typer.Typer()


# ---------- SUBFUNCTIONS ----------

def get_folder_labels(df: pd.DataFrame):
    """Return mapping of folders to labels for stratification."""
    all_folders = sorted([f for f in INTERIM_DATA_DIR.iterdir() if f.is_dir()])
    folder_labels = df.drop_duplicates(subset="folder")[["folder", "label"]]
    folder_labels = folder_labels[
        folder_labels["folder"].isin([f.name for f in all_folders])
    ]
    return folder_labels


def stratified_split(folder_labels: pd.DataFrame, seed=42):
    """Perform stratified split of folders into train, val, test."""
    train_folders, temp_folders = train_test_split(
        folder_labels, test_size=0.30, stratify=folder_labels["label"], random_state=seed
    )
    val_folders, test_folders = train_test_split(
        temp_folders, test_size=0.50, stratify=temp_folders["label"], random_state=seed
    )

    logger.success(
        f"Split sizes — Train: {len(train_folders)}, Val: {len(val_folders)}, Test: {len(test_folders)}"
    )
    return train_folders, val_folders, test_folders


def copy_folders(folders_df: pd.DataFrame, split_name: str):
    """Copy split folders into processed directory."""
    dest_dir = PROCESSED_DATA_DIR / split_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for _, row in folders_df.iterrows():
        folder_name = row["folder"]
        src = INTERIM_DATA_DIR / folder_name
        dst = dest_dir / folder_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    logger.success(f"Copied {len(folders_df)} folders into {split_name}/")


def create_balanced_train(train_folders: pd.DataFrame):
    """
    Create a balanced training set:
    - Keep all positive (label=1)
    - Sample negatives (label=0) at 1.5x number of positives
    """
    df_yes = train_folders[train_folders["label"] == 1]
    df_no = train_folders[train_folders["label"] == 0]

    n_yes = len(df_yes)
    n_no = int(np.floor(n_yes * 1.5))

    df_no_sampled = df_no.sample(n=n_no, random_state=42)
    balanced_df = (
        pd.concat([df_yes, df_no_sampled])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    bal_path = PROCESSED_DATA_DIR / "train_balanced"
    bal_path.mkdir(parents=True, exist_ok=True)

    for _, row in balanced_df.iterrows():
        folder_name = row["folder"]
        src_path = INTERIM_DATA_DIR / folder_name
        dst_path = bal_path / folder_name

        if src_path.exists():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            logger.warning(f"Source folder {src_path} does not exist")

    logger.success(
        f"Balanced train set created in {bal_path} with {len(balanced_df)} folders "
        f"(yes={n_yes}, no={n_no})"
    )


# ---------- TYPER COMMAND ----------

@app.command()
def split(
    annotations_file: Path = ANNOTATIONS_FILE,
    seed: int = 42,
    balanced: bool = False,
):
    """Generate train/val/test splits (and optionally a balanced train set)."""
    logger.info("Loading annotations...")
    df = pd.read_csv(annotations_file, sep="\t")

    folder_labels = get_folder_labels(df)
    train_folders, val_folders, test_folders = stratified_split(folder_labels, seed)

    # Copy folders into PROCESSED_DATA_DIR
    copy_folders(train_folders, "train")
    copy_folders(val_folders, "val")
    copy_folders(test_folders, "test")

    if balanced:
        create_balanced_train(train_folders)


if __name__ == "__main__":
    app()
