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

class ViTLSTM(nn.Module):
    def __init__(self, hidden_dim=256, lstm_layers=1, dropout=0.1):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        vit_feats = self.vit(x)
        vit_feats = vit_feats.view(B, T, -1)
        lstm_out, _ = self.lstm(vit_feats)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(1)

class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------
# Helpers
# ----------------------
def prepare_dataloaders(batch_size=4, max_seq_len=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = EmbryoSequenceDataset(PROCESSED_DATA_DIR / "train_balanced", transform, max_seq_len)
    val_ds = EmbryoSequenceDataset(PROCESSED_DATA_DIR / "val", transform, max_seq_len)
    test_ds = EmbryoSequenceDataset(PROCESSED_DATA_DIR / "test", transform, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10,
                patience=5, best_model_path="best_model.pth", scheduler=None, is_cnn=False, is_vit=False):
    best_val_f1 = 0.0
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_losses, all_preds, all_labels = [], [], []
        for inputs, labels in train_loader:
            if is_cnn:
                inputs = inputs.permute(0,2,1,3,4).to(device)
            else:
                inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1) if not is_vit else model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_preds, val_labels, val_losses = [], [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                if is_cnn:
                    inputs = inputs.permute(0,2,1,3,4).to(device)
                else:
                    inputs = inputs.to(device)
                labels = labels.float().to(device)
                outputs = model(inputs).squeeze(1) if not is_vit else model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds)
        if scheduler: scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            logger.success(f"Saved best model {best_model_path} F1={best_val_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.warning("Early stopping triggered")
                break

# ----------------------
# CLI
# ----------------------
@app.command()
def main(
    model_type: str = typer.Option("all", help="Model to train: ResNet, ViT, 3DCNN, or all"),
    batch_size: int = 4,
    num_epochs: int = 10,
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

    models_to_train = []
    if model_type.lower() in ["resnet", "all"]:
        models_to_train.append(("ResNet18LSTM", ResNet18LSTM(), False, False))
    if model_type.lower() in ["vit", "all"]:
        models_to_train.append(("ViTLSTM", ViTLSTM(), False, True))
    if model_type.lower() in ["3dcnn", "all"]:
        models_to_train.append(("3DCNN", Simple3DCNN(), True, False))

    for name, model, is_cnn, is_vit in models_to_train:
        logger.info(f"Training {name}...")
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        best_model_path = MODELS_DIR / f"{name}.pth"
        train_model(model, train_loader, val_loader, criterion, optimizer, device,
                    num_epochs=num_epochs, patience=patience, best_model_path=best_model_path,
                    scheduler=scheduler, is_cnn=is_cnn, is_vit=is_vit)
        logger.success(f"{name} training complete!")

if __name__ == "__main__":
    app()
