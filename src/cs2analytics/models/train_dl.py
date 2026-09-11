"""M8 §2 — training loop for the team-embedding net.

Leakage law, neural edition: `team_to_idx` is built from TRAIN only; teams first
seen in test map to index 0 (a real "<unk>" embedding vector — it only receives
gradients if the train set itself contains unknown-team rows, which it cannot,
so it stays at its random init and acts as a neutral vector).
Determinism: torch.manual_seed(42) + numpy seed; fixed batch order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cs2analytics.models.net import TeamEmbeddingNet

SEED = 42


def encode_teams(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Indices for train/test from a TRAIN-only vocabulary; unseen -> 0 (<unk>)."""
    teams = sorted(set(train_pairs["team1"]) | set(train_pairs["team2"]))
    team_to_idx: dict[str, int] = {t: i + 1 for i, t in enumerate(teams)}  # 0 = <unk>
    team_to_idx["<unk>"] = 0

    def enc(df: pd.DataFrame) -> np.ndarray:
        i1 = df["team1"].map(lambda t: team_to_idx.get(t, 0)).to_numpy()
        i2 = df["team2"].map(lambda t: team_to_idx.get(t, 0)).to_numpy()
        return np.stack([i1, i2], axis=1)

    return enc(train_pairs), enc(test_pairs), team_to_idx


def train_model(
    X_train_idx: np.ndarray,
    X_train_num: np.ndarray,
    y_train: np.ndarray,
    X_test_idx: np.ndarray,
    X_test_num: np.ndarray,
    y_test: np.ndarray,
    emb_dim: int = 16,
    hidden: int = 32,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 30,
    patience: int = 3,
) -> dict[str, Any]:
    """Train with early stop on train-loss plateau; return model + metrics."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    n_num = X_train_num.shape[1]
    n_teams = int(X_train_idx.max()) + 1
    model = TeamEmbeddingNet(n_teams=n_teams, emb_dim=emb_dim, hidden=hidden, n_num=n_num)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    idx_tr = torch.tensor(X_train_idx, dtype=torch.long)
    num_tr = torch.tensor(X_train_num, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)

    idx_te = torch.tensor(X_test_idx, dtype=torch.long)
    num_te = torch.tensor(X_test_num, dtype=torch.float32)

    history: list[float] = []
    best_loss = float("inf")
    plateau = 0
    n = len(y_train)
    for _epoch in range(max_epochs):
        model.train()
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            take = order[start : start + batch_size]
            opt.zero_grad()
            logits = model(idx_tr[take][:, 0], idx_tr[take][:, 1], num_tr[take])
            loss = loss_fn(logits, y_tr_t[take])
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * len(take)
        epoch_loss /= n
        history.append(epoch_loss)
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            plateau = 0
        else:
            plateau += 1
            if plateau >= patience:
                break

    model.eval()
    with torch.no_grad():
        val_logits = model(idx_te[:, 0], idx_te[:, 1], num_te)
        val_probs = torch.sigmoid(val_logits)
        train_logits = model(idx_tr[:, 0], idx_tr[:, 1], num_tr)
        train_probs = torch.sigmoid(train_logits)
    val_p = val_probs.numpy()
    train_p = train_probs.numpy()
    from cs2analytics.evaluation.metrics import accuracy_from_probs, brier_score, log_loss

    return {
        "model": model,
        "history": history,
        "logloss": log_loss(val_p, y_test),
        "brier": brier_score(val_p, y_test),
        "acc": accuracy_from_probs(val_p, y_test),
        "train_probs": train_p,
        "test_probs": val_p,
    }


def predict_proba_dl(model: nn.Module, X_idx: np.ndarray, X_num: np.ndarray) -> np.ndarray:
    """Probabilities from the trained net (sigmoid of logits)."""
    model.eval()
    with torch.no_grad():
        idx = torch.tensor(X_idx, dtype=torch.long)
        num = torch.tensor(X_num, dtype=torch.float32)
        logits = model(idx[:, 0], idx[:, 1], num)
        return torch.sigmoid(logits).numpy()
