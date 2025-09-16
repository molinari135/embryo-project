# Embryo Viability Prediction

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc] <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

  Classification of embryo images to predict viability using deep learning models (ResNet, ViT, LSTM, 3DCNN).
  The project follows the Cookiecutter Data Science structure and provides reproducible pipelines for preprocessing, annotation, feature extraction, training, and evaluation.

> A report and a presentation can be found in [docs folder](https://github.com/molinari135/embryo-project/tree/master/docs)

## Table of Contents
- [Project Organization](#project-organization)
- [Setup](#setup)
- [Data Structure](#data-structure)
- [Typer CLI Usage](#typer-cli-usage)
- [Models & Reports](#models--reports)
- [License](#license)

## Project Organization
```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         embryo_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
├── setup.cfg          <- Configuration file for flake8
└── embryo_project     <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes embryo_project a Python module
    ├── augment.py              <- Performs data augmentation
    ├── config.py               <- Store useful variables and configuration
    ├── dataset.py              <- Scripts to download or generate data
    ├── features.py             <- Code to create features for modeling
    ├── plots.py                <- Show and save plots
    ├── splits.py               <- Perform data splits in train, validation and test
    ├── modeling                
    │   ├── __init__.py 
    │   ├── evaluate_img.py     <- Code to run model inference with trained models          
    │   ├── evaluate.py         <- Code to run model inference with trained models          
    │   ├── train_img.py        <- Code to train models          
    │   └── train.py            <- Code to train models
    └── plots.py                <- Code to create visualizations
```

--------

## Setup

1. **Python Environment**
   - Python >= 3.13 required
   - Recommended: Create a virtual environment
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

## Data Structure
- Place raw embryo folders in `data/raw/`
- Annotation files (Excel/TSV) should be in `data/raw/` or `data/processed/`
- Preprocessed/interim/processed data will be saved in respective folders

## Typer CLI Usage

All main scripts use [Typer](https://typer.tiangolo.com/) for CLI commands. Run with:
```powershell
python -m embryo_project.<script> <command> [options]
```
Example:
```powershell
python -m embryo_project.dataset preprocess
```

### Available Typer Commands

#### Data Preprocessing
- `embryo_project/dataset.py`
  - `preprocess`: Run preprocessing pipeline on raw folders
    ```powershell
    python -m embryo_project.dataset preprocess --input-path data/raw --output-path data/interim
    ```

#### Annotation Preprocessing & Feature Extraction
- `embryo_project/features.py`
  - `preprocess`: Preprocess annotation Excel file
    ```powershell
    python -m embryo_project.features preprocess --input-path data/raw/annotations.xlsx --output-path data/processed/annotations.tsv
    ```

#### Train/Val/Test Split
- `embryo_project/splits.py`
  - (See script for available commands)

#### Model Training
- `embryo_project/modeling/train.py` (ResNet18LSTM)
  - `training`: Train LSTM model
    ```powershell
    python -m embryo_project.modeling.train training --model-name ResNet18LSTM --batch-size 64 --num-epochs 100 --patience 5 --lr 1e-4 --weight-decay 1e-4
    ```
- `embryo_project/modeling/train_img.py` (ResNet18Binary)
  - `training`: Train CNN model
    ```powershell
    python -m embryo_project.modeling.train_img training --model-name ResNet18Binary --batch-size 64 --num-epochs 100 --patience 5 --lr 1e-4 --weight-decay 1e-4
    ```

#### Model Evaluation
- `embryo_project/modeling/evaluate.py`
  - `main`: Evaluate LSTM model on test set
    ```powershell
    python -m embryo_project.modeling.evaluate main --model-name ResNet18LSTM --batch-size 64 --max-seq-len 20
    ```
- `embryo_project/modeling/evaluate_img.py`
  - `main`: Evaluate CNN model on test set
    ```powershell
    python -m embryo_project.modeling.evaluate_img main --model-name ResNet18Binary --batch-size 64
    ```

#### Plots
- `embryo_project/plots.py`
  - `plot_distribution`: Plot data distribution
    ```powershell
    python -m embryo_project.plots plot_distribution --data-dir data/processed --save --output-dir reports/figures
    ```

## Models & Reports
- Trained models are saved in `models/`
- Evaluation reports in `reports/`
- Figures in `reports/figures/`

## License
This project is licensed under [CC BY-NC 4.0][cc-by-nc].

For a copy of the license, please visit https://creativecommons.org/licenses/by-nc/4.0/

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]
