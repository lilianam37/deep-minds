"""Shared helpers for the Deep Minds phishing project (COSC 50523).

Every notebook imports from this one file, so every model is scored and
logged exactly the same way.

    import sys; sys.path.append(str(DATA_DIR))
    from project_utils import URL_FEATURES, set_seed, evaluate_model, log_result
"""
from __future__ import annotations

import json
import re
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# The 22 PhiUSIIL features that are computed from the URL string alone
# (no page download). The other 28 numeric columns need the web page.
URL_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex',
    'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
]


def set_seed(seed: int = 42) -> None:
    """Fix every random number generator we use so runs can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def evaluate_model(name: str, split: str, y_true, y_prob, threshold: float = 0.5,
                   show: bool = True, save_path: str | Path | None = None) -> dict:
    """Score one model on one split.

    y_prob = predicted probability of phishing (class 1) for each row.
    A row is predicted phishing when y_prob >= threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    results = {
        'Model': name,
        'Split': split,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'False positives': int(fp),   # legitimate sites flagged as phishing
        'False negatives': int(fn),   # phishing sites we missed
    }

    if show:
        print(pd.DataFrame([results]).round(4).to_string(index=False))
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array([[tn, fp], [fn, tp]]),
                                      display_labels=['Legitimate', 'Phishing'])
        disp.plot(cmap='Blues', values_format='d', colorbar=False)
        plt.title(f'{name} ({split})')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        plt.show()
    return results


def log_result(results: dict, hparams: dict, log_file: str | Path,
               use_wandb: bool = False, wandb_project: str = 'deep-minds-phishing') -> None:
    """Append one row (metrics + every hyperparameter) to the shared results log.

    The final hyperparameter table and ablation table are built from this file.
    """
    row = {'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'), **results,
           'Hyperparameters': json.dumps(hparams, default=str)}
    row.update({f'hp_{k}': (str(v) if isinstance(v, (list, tuple, dict)) else v)
                for k, v in hparams.items()})

    log_file = Path(log_file)
    new = pd.DataFrame([row])
    if log_file.exists():
        new = pd.concat([pd.read_csv(log_file), new], ignore_index=True)
    new.to_csv(log_file, index=False)

    if use_wandb:   # optional bonus: experiment tracking
        import wandb
        run = wandb.init(project=wandb_project, name=f"{results['Model']} | {results['Split']}",
                         config=hparams, reinit=True)
        run.log({k: v for k, v in results.items() if isinstance(v, (int, float))})
        run.finish()


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
MASTER_FILE = 'master_with_splits.csv'


def load_master(data_dir) -> pd.DataFrame:
    """The file made by week1/01_whitney_url_split: URL, label, split columns, 22 URL features."""
    return pd.read_csv(Path(data_dir) / MASTER_FILE, keep_default_na=False)


def get_feature_splits(df: pd.DataFrame, features=None, split_col: str = 'split'):
    """Return {'train': (X, y), 'val': (X, y), 'test': (X, y)} with scaled features.

    The scaler is fit on the TRAINING rows only (no leakage), then applied to val/test.
    """
    from sklearn.preprocessing import StandardScaler
    features = list(features or URL_FEATURES)
    scaler = StandardScaler().fit(df.loc[df[split_col] == 'train', features])
    data = {}
    for part in ['train', 'val', 'test']:
        rows = df[df[split_col] == part]
        data[part] = (scaler.transform(rows[features]).astype(np.float32),
                      rows['label'].to_numpy().astype(np.float32))
    return data, scaler


# ----------------------------------------------------------------------------
# URL -> numbers (for the character-level CNN)
# ----------------------------------------------------------------------------
_PREFIX = re.compile(r'^(https?://)?(www\.)?', flags=re.IGNORECASE)


def strip_prefix(url: str) -> str:
    """Ablation: remove 'http://', 'https://', 'www.' and a trailing '/'."""
    return _PREFIX.sub('', url.strip()).rstrip('/')


def build_vocab(train_urls) -> dict:
    """Give every character seen in the TRAINING URLs an ID. 0 = padding, 1 = unknown."""
    chars = sorted(set(''.join(train_urls)))
    return {c: i + 2 for i, c in enumerate(chars)}


def encode_urls(urls, vocab: dict, max_len: int) -> np.ndarray:
    """Each URL -> a row of max_len character IDs (cut if longer, padded with 0 if shorter)."""
    out = np.zeros((len(urls), max_len), dtype=np.int64)
    for i, url in enumerate(urls):
        ids = [vocab.get(c, 1) for c in url[:max_len]]
        out[i, :len(ids)] = ids
    return out


def get_url_sequences(df: pd.DataFrame, max_len: int = 200, split_col: str = 'split',
                      strip: bool = False):
    """Return ({'train': (X_ids, y), 'val': ..., 'test': ...}, vocab)."""
    urls = df['URL'].astype(str)
    if strip:
        urls = urls.map(strip_prefix)
    vocab = build_vocab(urls[df[split_col] == 'train'])
    data = {}
    for part in ['train', 'val', 'test']:
        mask = df[split_col] == part
        data[part] = (encode_urls(urls[mask].tolist(), vocab, max_len),
                      df.loc[mask, 'label'].to_numpy().astype(np.float32))
    return data, vocab


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # LR, RF, MLP, CNN (fixed order)


def plot_history(history: dict, title: str, save_path=None) -> None:
    """Training vs validation loss per epoch; dotted line = epoch kept by early stopping."""
    epochs = range(1, len(history['loss']) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(epochs, history['loss'], color=COLORS[0], lw=2, label='Training loss')
    ax.plot(epochs, history['val_loss'], color=COLORS[1], lw=2, label='Validation loss')
    if 'best_epoch' in history:
        ax.axvline(history['best_epoch'], color='#888888', ls=':', lw=1.5, label='Best epoch (kept)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Binary cross-entropy loss'); ax.set_title(title)
    ax.grid(axis='y', color='#e1e0d9', lw=0.8); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    plt.show()
