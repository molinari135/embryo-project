import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import typer
from loguru import logger

from embryo_project.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from train import EmbryoSequenceDataset, ResNet18LSTM, ViTLSTM, Simple3DCNN
from embryo_project import plots

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


def evaluate_model(model, test_loader, device, model_name, models_dir=MODELS_DIR, is_cnn=False):
    """Load model weights, run inference, and return predictions and labels."""
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

    return all_preds, all_labels


# ----------------------
# CLI
# ----------------------
@app.command()
def main(
    model_type: str = typer.Option("all", help="Model to evaluate: ResNet, ViT, 3DCNN, or all"),
    batch_size: int = 4,
    max_seq_len: int = 20
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

    results = {}
    for name, model, is_cnn in models_to_evaluate:
        logger.info(f"Evaluating {name}...")
        preds, labels = evaluate_model(model, test_loader, device, name, is_cnn=is_cnn)
        results[name] = {"preds": preds, "labels": labels}

        # Call plots.py function to save report and confusion matrix
        plots.plot_classification_results(preds, labels, name, FIGURES_DIR)

        logger.success(f"{name} evaluation complete!")

    return results


if __name__ == "__main__":
    app()
