"""Two fixed three-head diagnostic baselines; not an E0 or production protocol."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from facdigger.models.finance_patch_transformer import DateScoreOutput, LocalEncoderOutput

StatisticsKind = Literal["statistics_linear", "statistics_mlp"]


def statistics_dropout_identity(kind: str | None, dropout: float) -> dict[str, float]:
    """Old diagnostic identities implicitly mean the original hardcoded 0.1."""
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("statistics dropout must be finite and in [0, 1)")
    if kind != "statistics_mlp" and dropout != 0.1:
        raise ValueError("statistics dropout override requires statistics_mlp")
    return {"statistics_dropout": dropout} if dropout != 0.1 else {}


class FinanceStatisticsRanker(nn.Module):
    """Use the Finance date/loss loop with precomputed, non-learned encodings."""

    def __init__(
        self, kind: StatisticsKind, *, input_dim: int, horizons: tuple[int, ...],
        dropout: float = 0.1,
    ):
        super().__init__()
        statistics_dropout_identity(kind, dropout)
        self.horizons = horizons
        self.kind = kind
        if kind == "statistics_linear":
            self.head = nn.Linear(input_dim, len(horizons))
        elif kind == "statistics_mlp":
            self.head = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, len(horizons)),
            )
        else:
            raise ValueError("unknown diagnostic statistics model")

    def encode_local(self, values: torch.Tensor, observed_mask: torch.Tensor) -> LocalEncoderOutput:
        features = values[:, 0, :]
        return LocalEncoderOutput(features, features, features)

    def encode_market(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        return values[:, 0, :]

    def score_date(self, local: torch.Tensor, market: torch.Tensor) -> DateScoreOutput:
        features = torch.cat([local, market.expand(local.shape[0], -1)], dim=1)
        scores = self.head(features)
        return DateScoreOutput(
            scores, scores, scores, features, market, scores.new_zeros(len(self.horizons))
        )
