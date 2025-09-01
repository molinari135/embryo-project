import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
import typer

from embryo_project.config import PROCESSED_DATA_DIR, MODELS_DIR
from embryo_project.plots import plot_training_curves

app = typer.Typer()

# ----------------------
# Dataset
# ----------------------
class EmbryoImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # walk over all sequence folders
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            # now label each image individually from its filename
            image_files = [f for f in os.listdir(folder_path) if f.endswith(".JPG")]
            for img_file in image_files:
                label = 1 if img_file.endswith("_1.JPG") else 0
                self.samples.append((os.path.join(folder_path, img_file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ----------------------
# Model
# ----------------------
class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        # use the default pretrained weights
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_feats = resnet.fc.in_features
        resnet.fc = nn.Linear(num_feats, 1)  # binary classification
        self.model = resnet

    def forward(self, x):
        return self.model(x)

# ----------------------
# Helpers
# ----------------------
def prepare_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = EmbryoImageDataset(PROCESSED_DATA_DIR / "train", transform)
    val_ds   = EmbryoImageDataset(PROCESSED_DATA_DIR / "val", transform)
    test_ds  = EmbryoImageDataset(PROCESSED_DATA_DIR / "test", transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=10, patience=5, best_model_path="best_model.pth", scheduler=None,
                hyperparameters=None):

    best_val_f1 = 0.0
    epochs_no_improve = 0

    train_losses_per_epoch = []
    val_losses_per_epoch = []

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        epoch_train_losses = []
        all_preds, all_labels = [], []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_losses_per_epoch.append(sum(epoch_train_losses)/len(epoch_train_losses))

        # --- Validation ---
        model.eval()
        epoch_val_losses = []
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                epoch_val_losses.append(loss.item())
                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_losses_per_epoch.append(sum(epoch_val_losses)/len(epoch_val_losses))

        # --- Scheduler & Early stopping ---
        val_f1 = f1_score(val_labels, val_preds)
        if scheduler: scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hyperparameters": hyperparameters,
                "train_losses": train_losses_per_epoch,
                "val_losses": val_losses_per_epoch
            }, best_model_path)
            epochs_no_improve = 0
            logger.success(f"Saved best model {best_model_path} F1={best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.warning("Early stopping triggered")
                break

    plot_training_curves(
        train_losses=train_losses_per_epoch,
        val_losses=val_losses_per_epoch,
        model_name=Path(best_model_path).stem
    )

# ----------------------
# CLI
# ----------------------
@app.command()
def training(
    model_name: str = "ResNet18Binary",
    batch_size: int = 64,
    num_epochs: int = 100,
    patience: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4
):
    logger.info("Preparing datasets...")
    train_ds, _, _, train_loader, val_loader, _ = prepare_dataloaders(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    labels = [label for _, label in train_ds]
    counter = Counter(labels)
    pos_weight = torch.tensor([counter[0]/counter[1]]).to(device)

    os.makedirs(MODELS_DIR, exist_ok=True)

    hyperparams = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay
    }

    model = ResNet18Binary()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    best_model_path = MODELS_DIR / f"{model_name}.pth"

    logger.info(f"Training {model_name}...")
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        patience=patience,
        best_model_path=best_model_path,
        scheduler=scheduler,
        hyperparameters=hyperparams
    )

    logger.success(f"{model_name} training complete!")

if __name__ == "__main__":
    app()
