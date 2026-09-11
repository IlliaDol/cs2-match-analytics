"""M8 §1 — team-embedding network.

Architecture (spec): emb1, emb2 = E[i1], E[i2]; MLP input = [emb1 - emb2,
emb1 * emb2, numeric]; MLP = Linear(→hidden) → ReLU → Dropout(0.2) → Linear(→1).
Output = raw logits (BCEWithLogitsLoss applies the sigmoid).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TeamEmbeddingNet(nn.Module):
    """Logit = MLP([E[i1] - E[i2], E[i1] * E[i2], numeric]).

    Parameter count: n_teams*emb_dim + (emb_dim*2 + n_num)*hidden + hidden
    + hidden + 1 (two Linear layers, biases included).
    """

    def __init__(self, n_teams: int, emb_dim: int = 16, hidden: int = 32, n_num: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_teams, emb_dim)
        mlp_in = emb_dim * 2 + n_num
        self.fc1 = nn.Linear(mlp_in, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, team1_idx: torch.Tensor, team2_idx: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        e1 = self.embedding(team1_idx.long())
        e2 = self.embedding(team2_idx.long())
        diff = e1 - e2
        prod = e1 * e2
        x = torch.cat([diff, prod, numeric], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)