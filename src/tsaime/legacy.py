"""Backward-compatible scalar rolling API from ts-AIME 0.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .operators import scalar_aime_operator


@dataclass(frozen=True)
class RollingTSAIMEConfig:
    window: int
    n_permutations: int = 0
    null_method: Literal["periodic_shift", "seasonal_block"] = "periodic_shift"
    null_period: int = 1
    step: int = 1
    seed: int = 0
    alpha: float = 0.05

    def validate(self) -> None:
        if self.window < 3:
            raise ValueError("window must be at least 3")
        if self.n_permutations < 0:
            raise ValueError("n_permutations must be non-negative")
        if self.null_method not in {"periodic_shift", "seasonal_block"}:
            raise ValueError("unknown null_method")
        if self.null_period < 1 or self.step < 1:
            raise ValueError("null_period and step must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")


@dataclass
class TSAIMEResult:
    global_scores: pd.DataFrame
    local_contributions: pd.DataFrame
    null_lower: pd.DataFrame
    null_upper: pd.DataFrame
    p_values: pd.DataFrame
    significance: pd.DataFrame
    aligned: pd.DataFrame
    shifts: np.ndarray
    features: tuple[str, ...]
    config: RollingTSAIMEConfig

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "global": self.global_scores,
            "local": self.local_contributions,
            "env_lo": self.null_lower,
            "env_hi": self.null_upper,
            "pvals": self.p_values,
            "sig": self.significance,
            "aligned": self.aligned,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "config": asdict(self.config),
            "n_aligned_rows": len(self.aligned),
            "n_reported_windows": len(self.global_scores),
            "null_shifts": self.shifts.astype(int).tolist(),
        }


def rolling_correlation(
    left: Sequence[float], right: Sequence[float], window: int
) -> np.ndarray:
    """Return complete-window rolling Pearson correlations."""

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("left and right must be aligned vectors")
    if window < 3:
        raise ValueError("window must be at least 3")
    result = np.full(len(x), np.nan)
    if len(x) < window:
        return result
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = np.where(valid, x, 0.0)
    y_valid = np.where(valid, y, 0.0)

    def rolling_sum(values: np.ndarray) -> np.ndarray:
        cumulative = np.r_[0.0, np.cumsum(values, dtype=float)]
        return cumulative[window:] - cumulative[:-window]

    count = rolling_sum(valid.astype(float))
    sum_x = rolling_sum(x_valid)
    sum_y = rolling_sum(y_valid)
    numerator = rolling_sum(x_valid * y_valid) - sum_x * sum_y / np.maximum(count, 1)
    var_x = rolling_sum(x_valid * x_valid) - sum_x * sum_x / np.maximum(count, 1)
    var_y = rolling_sum(y_valid * y_valid) - sum_y * sum_y / np.maximum(count, 1)
    denominator = np.sqrt(np.maximum(var_x, 0) * np.maximum(var_y, 0))
    values = np.full_like(numerator, np.nan)
    usable = (count == window) & (denominator > 0)
    values[usable] = numerator[usable] / denominator[usable]
    result[window - 1 :] = np.clip(values, -1.0, 1.0)
    return result


def select_periodic_shifts(
    n: int,
    n_permutations: int,
    period: int,
    seed: int,
    *,
    minimum_phase_count: int | None = None,
) -> np.ndarray:
    """Draw nonidentity whole-period shifts in nominal row units."""

    if n < 2 or period < 1 or n_permutations < 0:
        raise ValueError("invalid shift configuration")
    if n_permutations == 0:
        return np.array([], dtype=int)
    count = minimum_phase_count if minimum_phase_count is not None else n // period
    if count < 2:
        raise ValueError("the aligned series is too short for a nonidentity periodic shift")
    candidates = period * np.arange(1, count, dtype=int)
    return np.random.default_rng(seed).choice(candidates, n_permutations, replace=True)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Return Benjamini-Hochberg rejection flags for one family."""

    p = np.asarray(p_values, dtype=float)
    rejected = np.zeros(len(p), dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return rejected
    values = p[finite]
    if np.any((values < 0) | (values > 1)):
        raise ValueError("finite p-values must lie in [0, 1]")
    order = np.argsort(values)
    passed = values[order] <= alpha * np.arange(1, len(values) + 1) / len(values)
    if passed.any():
        cutoff = values[order][np.flatnonzero(passed)[-1]]
        rejected[finite] = values <= cutoff
    return rejected


def _phase_groups(dates: pd.Series | None, n: int, period: int) -> tuple[np.ndarray, ...]:
    if dates is None:
        phase = np.arange(n) % period
    else:
        index = pd.DatetimeIndex(pd.to_datetime(dates))
        if len(index) != n or index.hasnans or not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("periodic nulls require unique increasing dates")
        if n == 1:
            phase = np.zeros(1, dtype=int)
        else:
            interval = int(np.gcd.reduce(np.diff(index.asi8)))
            elapsed = (index.asi8 - index.asi8[0]) // interval
            phase = elapsed % period
    return tuple(np.flatnonzero(phase == value) for value in np.unique(phase))


def _phase_shift(values: np.ndarray, groups: Sequence[np.ndarray], cycles: int) -> np.ndarray:
    shifted = np.empty_like(values)
    for positions in groups:
        if len(positions) < 2:
            raise ValueError("each phase needs at least two observations")
        shifted[positions] = np.roll(values[positions], cycles)
    return shifted


def _seasonal_runs(dates: pd.Series) -> tuple[np.ndarray, ...]:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    month = index.month.to_numpy()
    season = np.select(
        [np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8])],
        ["DJF", "MAM", "JJA"],
        default="SON",
    )
    year = index.year.to_numpy() + (month == 12)
    labels = np.asarray([f"{y}-{s}" for y, s in zip(year, season)])
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return tuple(np.split(np.arange(len(index)), boundaries))


class RollingTSAIME:
    """Backward-compatible rolling scalar-output ts-AIME estimator."""

    def __init__(self, config: RollingTSAIMEConfig):
        config.validate()
        self.config = config

    def fit(
        self,
        aligned: pd.DataFrame,
        features: Sequence[str],
        *,
        prediction_column: str = "Predictions",
        time_column: str = "Time",
        date_column: str | None = "Date",
    ) -> TSAIMEResult:
        feature_names = tuple(str(value) for value in features)
        required = [time_column, prediction_column, *feature_names]
        if date_column:
            required.append(date_column)
        missing = [column for column in required if column not in aligned]
        if missing:
            raise KeyError(f"aligned data is missing required columns: {missing}")
        if len(aligned) < self.config.window:
            raise ValueError("aligned data is shorter than the configured window")
        endpoints = np.arange(self.config.window - 1, len(aligned), self.config.step)
        identity = [time_column] + ([date_column] if date_column else [])
        base = aligned.iloc[endpoints][identity].reset_index(drop=True)
        global_scores = base.copy()
        local = base.copy()
        lower = base.copy()
        upper = base.copy()
        p_values = base.copy()
        forecast = aligned[prediction_column].to_numpy(float)
        shifts = np.array([], dtype=int)
        groups: tuple[np.ndarray, ...] = tuple()
        runs: tuple[np.ndarray, ...] = tuple()
        orders: list[np.ndarray] = []
        if self.config.n_permutations:
            if self.config.null_method == "periodic_shift":
                groups = _phase_groups(
                    aligned[date_column] if date_column else None,
                    len(aligned),
                    self.config.null_period,
                )
                shifts = select_periodic_shifts(
                    len(aligned),
                    self.config.n_permutations,
                    self.config.null_period,
                    self.config.seed,
                    minimum_phase_count=min(len(group) for group in groups),
                )
            else:
                if not date_column:
                    raise ValueError("seasonal_block nulls require a date column")
                runs = _seasonal_runs(aligned[date_column])
                if len(runs) < 2:
                    raise ValueError("seasonal_block nulls require at least two blocks")
                generator = np.random.default_rng(self.config.seed)
                identity_order = np.arange(len(runs))
                for _ in range(self.config.n_permutations):
                    order = generator.permutation(len(runs))
                    while np.array_equal(order, identity_order):
                        order = generator.permutation(len(runs))
                    orders.append(order)
        for feature in feature_names:
            x = aligned[feature].to_numpy(float)
            observed = rolling_correlation(x, forecast, self.config.window)[endpoints]
            global_scores[feature] = observed
            contributions = []
            for endpoint in endpoints:
                start = endpoint - self.config.window + 1
                _, endpoint_local = scalar_aime_operator(
                    aligned.iloc[start : endpoint + 1][list(feature_names)].to_numpy(float),
                    forecast[start : endpoint + 1],
                )
                contributions.append(endpoint_local[feature_names.index(feature)])
            local[feature] = contributions
            if not self.config.n_permutations:
                lower[feature] = np.nan
                upper[feature] = np.nan
                p_values[feature] = np.nan
                continue
            permuted = []
            for permutation in range(self.config.n_permutations):
                if self.config.null_method == "periodic_shift":
                    cycles = int(shifts[permutation]) // self.config.null_period
                    null_forecast = _phase_shift(forecast, groups, cycles)
                else:
                    null_forecast = forecast[np.concatenate([runs[index] for index in orders[permutation]])]
                permuted.append(rolling_correlation(x, null_forecast, self.config.window)[endpoints])
            null = np.asarray(permuted)
            lower[feature] = np.nanquantile(null, 0.025, axis=0)
            upper[feature] = np.nanquantile(null, 0.975, axis=0)
            p_values[feature] = (1 + np.sum(np.abs(null) >= np.abs(observed), axis=0)) / (
                1 + self.config.n_permutations
            )
        significance = base.copy()
        for feature in feature_names:
            significance[feature] = False
        for row in range(len(p_values)):
            flags = benjamini_hochberg(
                [p_values.at[row, feature] for feature in feature_names], self.config.alpha
            )
            for feature, flag in zip(feature_names, flags):
                significance.at[row, feature] = bool(flag)
        return TSAIMEResult(
            global_scores,
            local,
            lower,
            upper,
            p_values,
            significance,
            aligned.copy(),
            shifts,
            feature_names,
            self.config,
        )

