import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ADHDNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] = (256, 128, 64), dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int] = (256, 128, 64),
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[ADHDNet, list[dict]]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_val, dtype=torch.float32)
    y_te = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = ADHDNet(X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            logits = model(X_te.to(device))
            val_loss = criterion(logits, y_te.to(device)).item()
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            val_acc = (preds == y_val).mean()

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    return model, history


class EEGConvNet(nn.Module):

    def __init__(self, n_channels: int = 19, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).squeeze(-1)).squeeze(1)


def prepare_cnn_input(windows: np.ndarray) -> np.ndarray:
    return windows.transpose(0, 2, 1)


def train_cnn(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    n_channels: int = 19,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[EEGConvNet, list[dict]]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_tr = torch.tensor(X_train_raw, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_val_raw, dtype=torch.float32)
    y_te = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = EEGConvNet(n_channels=n_channels, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        train_loss /= len(X_train_raw)

        model.eval()
        with torch.no_grad():
            logits = model(X_te.to(device))
            val_loss = criterion(logits, y_te.to(device)).item()
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            val_acc = (preds == y_val).mean()

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    return model, history
