"""Calendar-preserving feature engineering and data-quality audit tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    """Half-open train, validation, and test boundaries."""

    start: pd.Timestamp | str
    train_end: pd.Timestamp | str
    validation_end: pd.Timestamp | str
    test_end: pd.Timestamp | str

    def timestamps(self) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        values = tuple(pd.Timestamp(value) for value in (
            self.start, self.train_end, self.validation_end, self.test_end
        ))
        if not all(left < right for left, right in zip(values, values[1:])):
            raise ValueError("split boundaries must be strictly increasing")
        return values


@dataclass(frozen=True)
class QualityGate:
    """Minimum usable rows and completeness for each chronological split."""

    minimum_rows: Mapping[str, int]
    minimum_coverage: float = 0.60

    def validate(self) -> None:
        required = {"train", "validation", "test"}
        if set(self.minimum_rows) != required:
            raise ValueError(f"minimum_rows must have exactly {sorted(required)}")
        if any(int(value) < 1 for value in self.minimum_rows.values()):
            raise ValueError("all minimum row counts must be positive")
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must lie in (0, 1]")


def complete_hourly(frame: pd.DataFrame, *, date_column: str = "Date") -> pd.DataFrame:
    """Reindex a table to its complete hourly calendar without interpolation."""

    if date_column not in frame:
        raise KeyError(f"missing date column: {date_column}")
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column]).sort_values(date_column)
    data = data.drop_duplicates(date_column, keep="last").set_index(date_column)
    if data.empty:
        return data.rename_axis(date_column).reset_index()
    calendar = pd.date_range(data.index.min().floor("h"), data.index.max().floor("h"), freq="h")
    return data.reindex(calendar).rename_axis(date_column).reset_index()


def prepare_chronological_pairs(
    frame: pd.DataFrame,
    state_columns: Sequence[str],
    target_columns: Sequence[str],
    split: ChronologicalSplit,
    *,
    max_target_horizon_hours: int,
    origin_step_hours: int = 1,
    date_column: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an already aligned supervised table without target leakage.

    The caller creates lagged inputs and future targets on a complete time grid.
    This generic function assigns source rows to chronological splits and ensures
    that the furthest target time remains inside the same split.
    """

    if origin_step_hours < 1:
        raise ValueError("origin_step_hours must be positive")
    if max_target_horizon_hours < 0:
        raise ValueError("max_target_horizon_hours must be non-negative")
    states = [str(value) for value in state_columns]
    targets = [str(value) for value in target_columns]
    if not states or not targets:
        raise ValueError("state_columns and target_columns must both be non-empty")
    required = [date_column, *states, *targets]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"missing aligned columns: {missing}")
    engineered = frame.copy()
    engineered[date_column] = pd.to_datetime(engineered[date_column], errors="coerce")
    engineered = (
        engineered.dropna(subset=[date_column])
        .sort_values(date_column)
        .drop_duplicates(date_column, keep="last")
    )
    start, train_end, validation_end, test_end = split.timestamps()
    epoch_hours = (
        (engineered[date_column] - pd.Timestamp("1970-01-01")) / pd.Timedelta(hours=1)
    ).round().astype("Int64")
    sampled = engineered.loc[(epoch_hours % origin_step_hours) == 0].copy()
    final_target_time = sampled[date_column] + pd.Timedelta(hours=max_target_horizon_hours)
    sampled["split"] = pd.NA
    sampled.loc[
        (sampled[date_column] >= start)
        & (sampled[date_column] < train_end)
        & (final_target_time < train_end),
        "split",
    ] = "train"
    sampled.loc[
        (sampled[date_column] >= train_end)
        & (sampled[date_column] < validation_end)
        & (final_target_time < validation_end),
        "split",
    ] = "validation"
    sampled.loc[
        (sampled[date_column] >= validation_end)
        & (sampled[date_column] < test_end)
        & (final_target_time < test_end),
        "split",
    ] = "test"
    candidates = sampled.dropna(subset=["split"]).copy()
    complete = candidates.dropna(subset=[*states, *targets]).copy()
    return candidates, complete


def _longest_missing_run(values: pd.Series) -> int:
    missing = values.isna().to_numpy(bool)
    if not missing.any():
        return 0
    padded = np.r_[False, missing, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(changes[1::2] - changes[::2]))


def audit_hourly_site(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    dataset: str = "",
    site: str = "",
    date_column: str = "Date",
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Summarize annual coverage and longest missing runs.

    Optional half-open ``start`` and ``end`` bounds keep audit denominators
    aligned with a prespecified study period.  This prevents a terminal
    boundary timestamp from being reported as a spurious extra year.
    """

    data = complete_hourly(frame, date_column=date_column)
    missing = [column for column in required_columns if column not in data]
    if missing:
        raise KeyError(f"missing audit columns: {missing}")
    observed_start = pd.Timestamp(data[date_column].min())
    observed_end = pd.Timestamp(data[date_column].max()) + pd.Timedelta(hours=1)
    audit_start = pd.Timestamp(start) if start is not None else observed_start
    audit_end = pd.Timestamp(end) if end is not None else observed_end
    if audit_end <= audit_start:
        raise ValueError("end must be later than start")

    # Reindex to the complete prespecified study calendar, not merely the
    # observed min/max range.  This makes boundary gaps count in annual
    # coverage and keeps expected_hours equal to the actual calendar hours.
    calendar = pd.date_range(
        audit_start,
        audit_end - pd.Timedelta(hours=1),
        freq="h",
    )
    data = (
        data.set_index(date_column)
        .reindex(calendar)
        .rename_axis(date_column)
        .reset_index()
    )
    data["year"] = data[date_column].dt.year
    rows: list[dict[str, float | int | str]] = []
    for year, group in data.groupby("year"):
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "site": site,
            "year": int(year),
            "expected_hours": int(len(group)),
        }
        for column in required_columns:
            row[f"{column}_coverage"] = float(group[column].notna().mean())
            row[f"{column}_max_gap_h"] = _longest_missing_run(group[column])
        row["all_required_coverage"] = float(
            group[list(required_columns)].notna().all(axis=1).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_quality_gate(
    candidates: pd.DataFrame,
    complete: pd.DataFrame,
    gate: QualityGate,
    *,
    dataset: str = "",
    site: str = "",
) -> dict[str, object]:
    """Evaluate fixed completeness rules for all chronological splits."""

    gate.validate()
    row: dict[str, object] = {"dataset": dataset, "site": site}
    reasons: list[str] = []
    for label in ("train", "validation", "test"):
        candidate_n = int((candidates["split"] == label).sum())
        usable_n = int((complete["split"] == label).sum())
        coverage = usable_n / candidate_n if candidate_n else 0.0
        row[f"{label}_candidate_n"] = candidate_n
        row[f"{label}_usable_n"] = usable_n
        row[f"{label}_coverage"] = coverage
        if usable_n < gate.minimum_rows[label] or coverage < gate.minimum_coverage:
            reasons.append(f"{label}: n={usable_n}, coverage={coverage:.3f}")
    row["quality_gate"] = "PASS" if not reasons else "FAIL"
    row["reason"] = "; ".join(reasons)
    return row
