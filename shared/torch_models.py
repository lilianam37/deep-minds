"""PyTorch models + one training loop shared by every notebook (MLP and char-CNN)."""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from project_utils import set_seed, get_url_sequences

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default settings. Every sweep/ablation changes ONE of these at a time.
DEFAULT_MLP = dict(hidden=(64, 32), dropout=0.0, lr=1e-3, batch_size=256,
                   max_epochs=20, patience=3)
DEFAULT_CNN = dict(max_len=200, embed_dim=32, filters=64, kernel_sizes=(3, 5, 7),
                   dropout=0.3, hidden=64, use_conv=True, strip_prefix=False,
                   lr=1e-3, batch_size=256, max_epochs=20, patience=3)


class MLP(nn.Module):
    """22 URL features -> Dense+ReLU layers -> 1 output (a logit; sigmoid gives P(phishing))."""

    def __init__(self, n_features: int, hidden=(64, 32), dropout: float = 0.0):
        super().__init__()
        layers, prev = [], n_features
        for units in hidden:
            layers += [nn.Linear(prev, units), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = units
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def features(self, x):          # last hidden layer (used for t-SNE)
        return self.body(x)

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)


class CharCNN(nn.Module):
    """Raw URL -> character IDs -> Embedding -> parallel Conv1d branches -> global max pool
    -> concatenate -> Dropout -> Dense+ReLU -> 1 output (logit).

    use_conv=False is the 'no convolution' ablation: Embedding -> global max pool -> Dense.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, filters: int = 64,
                 kernel_sizes=(3, 5, 7), dropout: float = 0.3, hidden: int = 64,
                 use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if use_conv:
            self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, k) for k in kernel_sizes])
            pooled_size = filters * len(kernel_sizes)
        else:
            pooled_size = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(pooled_size, hidden)
        self.out = nn.Linear(hidden, 1)

    def features(self, x):
        e = self.embedding(x)                       # (batch, length, embed_dim)
        if self.use_conv:
            e = e.transpose(1, 2)                   # Conv1d wants (batch, channels, length)
            z = torch.cat([torch.relu(conv(e)).max(dim=2).values for conv in self.convs], dim=1)
        else:
            z = e.max(dim=1).values
        return torch.relu(self.fc(self.dropout(z)))

    def forward(self, x):
        return self.out(self.features(x)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _loader(X, y, batch_size, shuffle, x_dtype):
    ds = TensorDataset(torch.as_tensor(X, dtype=x_dtype), torch.as_tensor(y, dtype=torch.float32))
    gen = torch.Generator().manual_seed(42) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def train_model(model, X_train, y_train, X_val, y_val, lr=1e-3, batch_size=256,
                max_epochs=20, patience=3, x_dtype=torch.float32, verbose=True) -> dict:
    """Adam + binary cross-entropy, with early stopping on validation loss.

    Keeps the weights from the epoch with the lowest validation loss.
    Returns a history dict: loss, val_loss per epoch, and best_epoch.
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()       # sigmoid + binary cross-entropy in one, numerically stable
    train_dl = _loader(X_train, y_train, batch_size, True, x_dtype)
    val_dl = _loader(X_val, y_val, 1024, False, x_dtype)

    history = {'loss': [], 'val_loss': []}
    best_loss, best_state, best_epoch, wait = float('inf'), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        train_loss = total / len(train_dl.dataset)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += loss_fn(model(xb), yb).item() * len(yb)
        val_loss = total / len(val_dl.dataset)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if verbose:
            print(f'Epoch {epoch:2d}  train loss {train_loss:.5f}  val loss {val_loss:.5f}')

        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f'Early stopping: no improvement for {patience} epochs.')
                break

    model.load_state_dict(best_state)
    history['best_epoch'] = best_epoch
    return history


@torch.no_grad()
def predict_proba(model, X, x_dtype=torch.float32, batch_size=1024) -> np.ndarray:
    """P(phishing) for every row."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=x_dtype).to(DEVICE)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def hidden_features(model, X, x_dtype=torch.float32, batch_size=1024) -> np.ndarray:
    """Last hidden layer output for every row (for the t-SNE plot)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=x_dtype).to(DEVICE)
        out.append(model.features(xb).cpu().numpy())
    return np.concatenate(out)


def fit_mlp(data: dict, config: dict | None = None, verbose=True):
    """Train an MLP on get_feature_splits() output. Returns (model, history, config)."""
    cfg = {**DEFAULT_MLP, **(config or {})}
    set_seed(42)
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    model = MLP(X_train.shape[1], hidden=cfg['hidden'], dropout=cfg['dropout'])
    history = train_model(model, X_train, y_train, X_val, y_val, lr=cfg['lr'],
                          batch_size=cfg['batch_size'], max_epochs=cfg['max_epochs'],
                          patience=cfg['patience'], verbose=verbose)
    return model, history, cfg


def fit_char_cnn(df, config: dict | None = None, split_col='split', verbose=True):
    """Encode URLs and train a CharCNN. Returns (model, history, cfg, seqs, vocab)."""
    cfg = {**DEFAULT_CNN, **(config or {})}
    seqs, vocab = get_url_sequences(df, cfg['max_len'], split_col, cfg['strip_prefix'])
    set_seed(42)
    model = CharCNN(len(vocab) + 2, embed_dim=cfg['embed_dim'], filters=cfg['filters'],
                    kernel_sizes=cfg['kernel_sizes'], dropout=cfg['dropout'],
                    hidden=cfg['hidden'], use_conv=cfg['use_conv'])
    X_train, y_train = seqs['train']
    X_val, y_val = seqs['val']
    history = train_model(model, X_train, y_train, X_val, y_val, lr=cfg['lr'],
                          batch_size=cfg['batch_size'], max_epochs=cfg['max_epochs'],
                          patience=cfg['patience'], x_dtype=torch.long, verbose=verbose)
    return model, history, cfg, seqs, vocab
