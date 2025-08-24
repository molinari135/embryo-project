# embryo_project/modeling/predict.py

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from loguru import logger
import typer

from embryo_project.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
from train import EmbryoSequenceDataset, ResNet18LSTM, ViTLSTM, Simple3DCNN

app = typer.Typer()

# ----------------------
# Helpers
# ----------------------
def prepare_test_loader(batch_size=4, max_seq_len=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_ds = EmbryoSequenceDataset(PROCESSED_DATA_DIR / "test", transform, max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader


def evaluate_and_save_results(model, test_loader, device, model_name, models_dir=MODELS_DIR,
                              reports_dir=REPORTS_DIR, figures_dir=FIGURES_DIR, is_cnn=False,
                              report_name=None):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, f"{model_name}.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            if is_cnn:
                inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
            else:
                inputs = inputs.to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds,
                                   target_names=["Class 0", "Class 1"], digits=4)

    # Use custom report name if provided
    if report_name is None:
        report_filename = f"{model_name}_report.txt"
    else:
        report_filename = report_name if report_name.endswith(".txt") else f"{report_name}.txt"

    report_path = os.path.join(reports_dir, report_filename)
    with open(report_path, "w") as f:
        f.write(report)
    logger.success(f"Classification report saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    cm_path = os.path.join(figures_dir, f"{model_name}_cm.png")
    plt.savefig(cm_path)
    plt.close()
    logger.success(f"Confusion matrix saved to {cm_path}")


# ----------------------
# CLI
# ----------------------
@app.command()
def main(
    model_type: str = typer.Option("all", help="Model to evaluate: ResNet, ViT, 3DCNN, or all"),
    batch_size: int = 4,
    max_seq_len: int = 20,
    report_name: str = typer.Option(None, help="Custom name for the classification report .txt file")
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_loader = prepare_test_loader(batch_size=batch_size, max_seq_len=max_seq_len)

    models_to_evaluate = []
    if model_type.lower() in ["resnet", "all"]:
        models_to_evaluate.append(("ResNet18LSTM", ResNet18LSTM(), False))
    if model_type.lower() in ["vit", "all"]:
        models_to_evaluate.append(("ViTLSTM", ViTLSTM(), False))
    if model_type.lower() in ["3dcnn", "all"]:
        models_to_evaluate.append(("3DCNN", Simple3DCNN(), True))

    for name, model, is_cnn in models_to_evaluate:
        logger.info(f"Evaluating {name}...")
        evaluate_and_save_results(model, test_loader, device, name, is_cnn=is_cnn, report_name=report_name)
        logger.success(f"{name} evaluation complete!")


if __name__ == "__main__":
    app()
