from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer, re, shutil

from embryo_project.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def to_long_path(path: Path) -> str:
    return r"\\?\{}".format(str(path.resolve()))


# ---------- SUBFUNCTIONS ----------

def extract_tmp_dir(input_dir: Path, output_dir: Path):
    """Extract subdirectories from a fixed path into interim data dir."""
    fix_dir = input_dir / Path(
        "2022_CAMPIONATO/17.08.2022_CAMPIONATO/D2022.08.17_S00149_I4203_P_WELL01_CAMPIONATO"
    )
    work_dir = output_dir / "tmp"
    tmp_dir = work_dir / (fix_dir.name + "_tmp")

    shutil.copytree(fix_dir, tmp_dir)

    for subfolder in tmp_dir.iterdir():
        if subfolder.is_dir():
            dest = output_dir / subfolder.name
            logger.info(f"Moving {subfolder.name} -> {dest}")
            shutil.move(str(subfolder), str(dest))

    shutil.rmtree(tmp_dir)


def copy_datasets(input_dir: Path, output_dir: Path):
    """Copy all D202X directories into interim dir, handling duplicates."""
    well_name_pattern = re.compile(r"^D202[0-3].*WELL\d{1,2}")
    found, copied, conflicts = 0, 0, 0

    for folder in input_dir.rglob("*"):
        if folder.is_dir() and well_name_pattern.match(folder.name):
            found += 1
            dest = output_dir / folder.name
            if not dest.exists():
                shutil.copytree(to_long_path(folder), to_long_path(dest))
                copied += 1
            else:
                conflicts += 1

    logger.info(f"Total number of folders found: {found}")
    logger.success(f"Copied folders: {copied}")
    logger.warning(f"Conflicts (already copied): {conflicts}")


def remove_empty_dirs(output_dir: Path):
    """Remove empty directories inside interim dir."""
    empty_dirs = 0
    for folder in output_dir.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            empty_dirs += 1
            logger.info(f"Removed empty folder: {folder.name}")
    logger.success(f"Removed folders: {empty_dirs}")


def rename_directories(output_dir: Path):
    """Standardize directory names inside interim dir."""
    dir_name_pattern = r"^(.*?WELL\d{1,2})"
    pattern_matched, renamed_dirs = 0, 0

    for folder in output_dir.iterdir():
        if not folder.is_dir():
            continue

        relative_str = str(folder.relative_to(output_dir))
        match = re.match(dir_name_pattern, relative_str)

        if match:
            clean_name = match.group(1)
            new_path = folder.parent / clean_name

            if folder.name != clean_name:
                if not new_path.exists():
                    folder.rename(new_path)
                    renamed_dirs += 1
                else:
                    logger.warning(f"{new_path} already exists.")

            pattern_matched += 1
        else:
            logger.warning(f"No match for {relative_str}")

    logger.info(f"Total directories: {len(list(output_dir.iterdir()))}")
    logger.success(f"Matched: {pattern_matched}, Renamed: {renamed_dirs}")


def fix_specific_names(output_dir: Path):
    """Fix specific directories that don't follow the convention."""
    old_to_new = {
        "D2022.03.02_S00116_I4203_P_WELL01": "D2022.03.02_S00116_I4203_P_WELL10",
        "D2022.03.02_S00116_I4203_P_WELL01_CAMPIONATO": "D2022.03.02_S00116_I4203_P_WELL01",
    }

    for old_name, new_name in old_to_new.items():
        old_path = output_dir / old_name
        new_path = output_dir / new_name

        if old_path.exists():
            if not new_path.exists():
                old_path.rename(new_path)
                logger.success(f"Renamed: {old_name} -> {new_name}")
            else:
                logger.warning(f"{new_name} already exists.")
        else:
            logger.warning(f"Folder not found: {old_name}")


def fix_well_suffix(output_dir: Path):
    """Fix directories ending with WELL1 instead of WELL01."""
    for folder in output_dir.rglob("*"):
        if folder.is_dir() and folder.name.endswith("WELL1"):
            new_name = folder.name.replace("WELL1", "WELL01")
            new_path = folder.parent / new_name
            logger.info(f"Renaming: {folder} -> {new_path}")
            folder.rename(new_path)


# ---------- PIPELINE ----------

def folder_preprocessing(input_dir: Path, output_dir: Path):
    logger.info("Processing input directory...")
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_tmp_dir(input_dir, output_dir)
    copy_datasets(input_dir, output_dir)
    remove_empty_dirs(output_dir)
    rename_directories(output_dir)
    fix_specific_names(output_dir)
    fix_well_suffix(output_dir)


# ---------- TYPER COMMANDS ----------

@app.command()
def preprocess(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
):
    """Run the preprocessing pipeline on raw dataset folders."""
    folder_preprocessing(input_path, output_path)


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
):
    logger.info("Running preprocessing pipeline...")
    folder_preprocessing(input_path, output_path)

    # Example extra processing with tqdm
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")

    logger.success("Dataset processing complete.")
