# embryo_project/modeling/train.py

import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import vit_b_16
from PIL import Image
from loguru import logger
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
import typer

from embryo_project.config import PROCESSED_DATA_DIR, MODELS_DIR
from embryo_project.plots import plot_training_curves

app = typer.Typer()

# ----------------------
# Dataset
# ----------------------
class EmbryoSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_seq_len=20):
        self.root_dir = root_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        folder_path = os.path.join(self.root_dir, folder)

        image_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".JPG")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        images = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        if self.max_seq_len:
            images = images[:self.max_seq_len]
            while len(images) < self.max_seq_len:
                images.append(torch.zeros_like(images[0]))

        images_tensor = torch.stack(images)
        label = 1 if any(f.endswith("_1.JPG") for f in image_files) else 0
        return images_tensor, label

# ----------------------
# Models
# ----------------------
class ResNet18LSTM(nn.Module):
    def __init__(self, cnn_embed_dim=512, lstm_hidden_size=128, num_layers=1, bidirectional=True):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_embed_dim = cnn_embed_dim
        self.lstm = nn.LSTM(
            input_size=cnn_embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(lstm_hidden_size * direction_factor, 1)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        cnn_feats = self.cnn(x).view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_feats)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)


# ----------------------
# Helpers
# ----------------------
def prepare_dataloaders(batch_size=4, max_seq_len=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # TODO add custom train path
    train_ds = EmbryoSequenceDataset(PROCESSED_DATA_DIR / "train", transform, max_seq_len)
    val_ds = EmbryoSequenceDataset(PROCESSED_DATA_DIR / "val", transform, max_seq_len)
    test_ds = EmbryoSequenceDataset(PROCESSED_DATA_DIR / "test", transform, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10,
                patience=5, best_model_path="best_model.pth", scheduler=None,
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

    # --- Plot training curves ---
    plot_training_curves(
        train_losses=train_losses_per_epoch,
        val_losses=val_losses_per_epoch,
        model_name=best_model_path.stem
    )


# ----------------------
# CLI
# ----------------------
@app.command()
def training(
    model_name: str = "ResNet18LSTM",
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

    model = ResNet18LSTM()

    logger.info(f"Training {model_name}...")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    best_model_path = MODELS_DIR / f"{model_name}.pth"

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