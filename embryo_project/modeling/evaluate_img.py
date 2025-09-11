import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import typer
from loguru import logger

from embryo_project.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from embryo_project.modeling.train_img import EmbryoImageDataset, ResNet18Binary
from embryo_project.plots import plot_classification_results

app = typer.Typer()

# ----------------------
# Helpers
# ----------------------
def prepare_test_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_ds = EmbryoImageDataset(PROCESSED_DATA_DIR / "test", transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader


def evaluate_model(model, test_loader, device, model_name, models_dir=MODELS_DIR):
    """Load model weights, run inference, and return predictions and labels."""
    best_model_path = os.path.join(models_dir, f"{model_name}.pth")
    checkpoint = torch.load(best_model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            outputs = model(inputs).squeeze(1)   # shape [B]
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


# ----------------------
# CLI
# ----------------------
@app.command()
def main(
    model_name: str = "ResNet18Binary",
    batch_size: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_loader = prepare_test_loader(batch_size=batch_size)

    results = {}
    model = ResNet18Binary()

    logger.info(f"Evaluating {model_name}...")
    preds, labels = evaluate_model(model, test_loader, device, model_name)
    results[model_name] = {"preds": preds, "labels": labels}

    plot_classification_results(preds, labels, model_name, FIGURES_DIR)

    logger.success(f"{model_name} evaluation complete!")

    return results


if __name__ == "__main__":
    app()
