from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer

from embryo_project.config import RAW_DATA_DIR, ANNOTATIONS_FILE

app = typer.Typer()


# ---------- SUBFUNCTIONS ----------

def load_annotations(file_path: Path) -> pd.DataFrame:
    """Load annotations Excel file."""
    logger.info(f"Loading annotations from {file_path}...")
    return pd.read_excel(file_path)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    return df.rename(columns={
        "Anno": "year",
        "Nome cartella": "folder",
        "etichetta": "label",
        "foto \"cruciale\" ": "image",
        "tot elementi": "elements",
        "commenti": "comments"
    })


def clean_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing folder or image."""
    original_rows = df.shape[0]

    df = df.dropna(subset=["folder"])
    df = df.dropna(subset=["image"])

    cleaned_rows = df.shape[0]
    logger.success(f"Removed {original_rows - cleaned_rows} NaN rows")
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map labels 'blastocisti si/no' to 1/0."""
    df["label"] = df["label"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip().lower() == "blastocisti si"
        else 0 if isinstance(x, str) and x.strip().lower() == "blastocisti no"
        else x
    )
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns."""
    return df.drop(["#", "year", "elements", "comments"], axis=1, errors="ignore")


def save_annotations(df: pd.DataFrame, output_path: Path):
    """Save preprocessed annotations to file."""
    df.to_csv(output_path, sep="\t", index=False)
    logger.success(f"The file has been saved in {output_path}")


# ---------- PIPELINE ----------

def annotation_preprocessing(input_path: Path, output_path: Path):
    """Run the full preprocessing pipeline on annotations."""
    df = load_annotations(input_path)
    df = rename_columns(df)
    df = clean_annotations(df)
    df = drop_unused_columns(df)
    df = map_labels(df)
    save_annotations(df, output_path)


# ---------- TYPER COMMANDS ----------

@app.command()
def preprocess(
    input_path: Path = RAW_DATA_DIR / "annotations.xlsx",
    output_path: Path = ANNOTATIONS_FILE,
):
    """Run preprocessing pipeline on annotations."""
    annotation_preprocessing(input_path, output_path)


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "annotations.xlsx",
    output_path: Path = ANNOTATIONS_FILE,
):
    logger.info("Running annotation preprocessing...")
    annotation_preprocessing(input_path, output_path)

    # Example: feature extraction step
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
