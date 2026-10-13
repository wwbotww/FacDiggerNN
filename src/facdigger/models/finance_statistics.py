"""Fixed statistics diagnostics; not an E0 or production protocol."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from facdigger.models.finance_patch_transformer import DateScoreOutput, LocalEncoderOutput

StatisticsKind = Literal[
    "statistics_linear",
    "statistics_mlp",
    "statistics_conditional_linear",
]


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
        self,
        kind: StatisticsKind,
        *,
        input_dim: int,
        horizons: tuple[int, ...],
        dropout: float = 0.1,
        local_dim: int | None = None,
        market_channels: int | None = None,
    ):
        super().__init__()
        statistics_dropout_identity(kind, dropout)
        self.horizons = horizons
        self.kind = kind
        if kind in {"statistics_linear", "statistics_conditional_linear"}:
            self.head = nn.Linear(input_dim, len(horizons))
            if kind == "statistics_conditional_linear":
                if (
                    local_dim is None
                    or market_channels is None
                    or local_dim <= 0
                    or market_channels <= 0
                    or input_dim - local_dim < market_channels
                ):
                    raise ValueError(
                        "conditional Linear requires local and current market dimensions"
                    )
                self.market_channels = market_channels
                # Do not consume RNG: the base head and subsequent random stream stay paired.
                self.interaction = nn.Parameter(
                    torch.zeros(len(horizons), market_channels, local_dim)
                )
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
        if self.kind == "statistics_conditional_linear":
            # The dataset must verify these are t-day values, not last-known observations.
            current = market[:, -self.market_channels :]
            coefficients = torch.einsum("bk,hkl->bhl", current, self.interaction)[0]
            scores = scores + torch.nn.functional.linear(local, coefficients)
        return DateScoreOutput(
            scores, scores, scores, features, market, scores.new_zeros(len(self.horizons))
        )
