"""Date-aware delivery identity, separate from the model's computational universe."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import Field, model_validator

from facdigger.data.config import StrictModel
from facdigger.data.contracts import DataContractError

CANDIDATE_COLUMNS = ["security_id", "symbol", "asof_date", "eligible"]
_ISIN_ID = re.compile(r"^eodhd:isin:[A-Z]{2}[A-Z0-9]{9}[0-9]$")


class DeliveryTarget(StrictModel):
    instrument_id: str = Field(min_length=1)
    active_from: date | None = None
    active_to: date | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> DeliveryTarget:
        if self.active_from and self.active_to and self.active_from > self.active_to:
            raise ValueError("target active_from must not exceed active_to")
        return self


class DeliveryIdentity(StrictModel):
    instrument_id: str = Field(min_length=1)
    security_id: str = Field(min_length=1)
    source_security_id: str | None = Field(default=None, min_length=1)
    valid_from: date
    valid_to: date
    evidence: str = Field(min_length=1)

    @property
    def source_id(self) -> str:
        return self.source_security_id or self.security_id

    @model_validator(mode="after")
    def validate_identity(self) -> DeliveryIdentity:
        if self.valid_from > self.valid_to:
            raise ValueError("identity valid_from must not exceed valid_to")
        if not self.evidence.strip():
            raise ValueError("identity evidence must describe the verified mapping")
        if (
            self.source_id.startswith("eodhd:isin:")
            and self.security_id.startswith("eodhd:isin:")
            and self.source_id != self.security_id
        ):
            raise ValueError("do not rewrite a historical ISIN as a different ISIN")
        for value in (self.source_id, self.security_id):
            if value.startswith("eodhd:isin:") and _ISIN_ID.fullmatch(value) is None:
                raise ValueError("invalid EODHD ISIN identity format")
        return self


class DeliveryConfig(StrictModel):
    """A consumer-confirmed target list and independent, inclusive identity intervals."""

    targets: list[DeliveryTarget] = Field(min_length=1)
    identities: list[DeliveryIdentity] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bindings(self) -> DeliveryConfig:
        targets = [target.instrument_id for target in self.targets]
        if len(set(targets)) != len(targets):
            raise ValueError("delivery targets must be unique")
        for identity in self.identities:
            if identity.instrument_id not in targets:
                raise ValueError("identity refers to an undeclared delivery target")
        # Any overlap affecting a target, source or wire identity is ambiguous.
        for key in ("instrument_id", "source_id", "security_id"):
            groups: dict[str, list[DeliveryIdentity]] = {}
            for identity in self.identities:
                groups.setdefault(getattr(identity, key), []).append(identity)
            for entries in groups.values():
                ordered = sorted(entries, key=lambda value: value.valid_from)
                if any(
                    a.valid_to >= b.valid_from
                    for a, b in zip(ordered, ordered[1:], strict=False)
                ):
                    raise ValueError(f"overlapping delivery identity intervals for {key}")
        return self


def load_delivery_config(path: str | Path) -> DeliveryConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("delivery configuration root must be a mapping")
    return DeliveryConfig.model_validate(raw)


def delivery_identity_policy(frame: pl.DataFrame) -> Literal[
    "provider_neutral_security_id", "eodhd_isin_only"
]:
    ids = frame["security_id"].unique().to_list()
    return (
        "eodhd_isin_only"
        if ids and all(_ISIN_ID.fullmatch(value) for value in ids)
        else "provider_neutral_security_id"
    )


@dataclass(frozen=True)
class DeliverySelection:
    source_candidates: pl.DataFrame
    candidates: pl.DataFrame
    bindings: pl.DataFrame | None
    audit: dict[str, Any] | None

    def for_dates(self, dates: list[date]) -> DeliverySelection:
        return DeliverySelection(
            self.source_candidates.filter(pl.col("asof_date").is_in(dates)),
            self.candidates.filter(pl.col("asof_date").is_in(dates)),
            None if self.bindings is None else self.bindings.filter(
                pl.col("asof_date").is_in(dates)
            ),
            None,
        )

    def project_scores(self, scores: pl.DataFrame) -> pl.DataFrame:
        """Project already-complete model scores; never recompute or rerank them."""
        selected = scores.join(
            self.source_candidates.select("security_id", "asof_date"),
            on=["security_id", "asof_date"], how="semi",
        )
        if self.bindings is not None:
            selected = selected.rename({"security_id": "source_security_id"}).join(
                self.bindings.select("source_security_id", "asof_date", "security_id"),
                on=["source_security_id", "asof_date"], how="inner", validate="1:1",
            )
        return selected.select("security_id", "asof_date", "score").sort(
            "asof_date", "security_id"
        )


def require_resolved_delivery_identities(
    selection: DeliverySelection, unscorable: list[dict[str, Any]],
) -> None:
    """A mapped but unresolved source identity is not an ordinary missing score."""
    selected = {(identity, str(day)) for identity, day in
                selection.source_candidates.select("security_id", "asof_date").iter_rows()}
    unresolved = sorted({row["security_id"] for row in unscorable
                         if row["reason"] == "unresolved_security_identity"
                         and (row["security_id"], str(row["asof_date"])) in selected})
    if unresolved:
        raise DataContractError("delivery identity is unresolved: " + ", ".join(unresolved))


def resolve_delivery(
    candidates: pl.DataFrame,
    config: DeliveryConfig | None,
    *,
    eligible_only: bool = False,
) -> DeliverySelection:
    """Resolve the declared target/date matrix before checking score coverage."""
    candidates = candidates.select(CANDIDATE_COLUMNS).sort("asof_date", "security_id")
    if candidates.is_empty() or candidates.null_count().sum_horizontal().item():
        raise DataContractError("delivery candidate universe is empty or contains nulls")
    if candidates.select("security_id", "asof_date").n_unique() != candidates.height:
        raise DataContractError("delivery candidate universe has duplicate keys")
    if config is None:
        # Generic research deliveries remain supported, without claiming consumer
        # identity verification. Provider-symbol IDs require an explicit mapping.
        selected = candidates.filter(pl.col("eligible")) if eligible_only else candidates
        if selected.filter(pl.col("security_id").str.starts_with("eodhd:symbol:")).height:
            raise DataContractError(
                "delivered provider-symbol identities require a delivery profile"
            )
        return DeliverySelection(selected, selected, None, None)

    dates = candidates.select("asof_date").unique()
    targets = pl.DataFrame([
        {
            "instrument_id": target.instrument_id,
            "active_from": target.active_from or date.min,
            "active_to": target.active_to or date.max,
        }
        for target in config.targets
    ])
    expected = dates.join(targets, how="cross").filter(
        pl.col("asof_date").is_between(pl.col("active_from"), pl.col("active_to"))
    ).select("instrument_id", "asof_date")
    identities = pl.DataFrame([
        {
            "instrument_id": binding.instrument_id,
            "source_security_id": binding.source_id,
            "security_id": binding.security_id,
            "valid_from": binding.valid_from,
            "valid_to": binding.valid_to,
        }
        for binding in config.identities
    ])
    bindings = expected.join(identities, on="instrument_id", how="inner").filter(
        pl.col("asof_date").is_between(pl.col("valid_from"), pl.col("valid_to"))
    )
    missing = expected.join(
        bindings.select("instrument_id", "asof_date"),
        on=["instrument_id", "asof_date"], how="anti",
    )
    if missing.height:
        raise DataContractError(
            f"delivery identity is missing or expired: {missing.head(5).to_dicts()}"
        )
    joined = bindings.join(
        candidates.rename({"security_id": "source_security_id"}),
        on=["source_security_id", "asof_date"], how="left", validate="1:1",
    )
    absent = joined.filter(pl.col("eligible").is_null())
    if absent.height:
        raise DataContractError(
            "declared delivery targets are absent from the candidate universe: "
            f"{absent.select('instrument_id', 'asof_date').head(5).to_dicts()}"
        )
    available = joined.filter(pl.col("eligible").is_not_null())
    ineligible_count = available.filter(~pl.col("eligible")).height
    if eligible_only:
        available = available.filter(pl.col("eligible"))
    selected = available.select(CANDIDATE_COLUMNS).sort("asof_date", "security_id")
    source_candidates = available.select(
        pl.col("source_security_id").alias("security_id"), "symbol", "asof_date", "eligible"
    ).sort("asof_date", "security_id")
    audit = {
        "delivery": config.model_dump(mode="json"),
        "computational_candidate_rows": candidates.height,
        "computational_eligible_rows": int(candidates["eligible"].sum()),
        "requested_target_rows": expected.height,
        "absent_candidate_rows": absent.height,
        "ineligible_candidate_rows": ineligible_count,
        "delivered_candidate_rows": selected.height,
        "delivered_eligible_rows": int(selected["eligible"].sum()),
    }
    return DeliverySelection(source_candidates, selected, available, audit)
