"""Time-series performance metrics and dependence-preserving uncertainty tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .operators import fit_inverse_operator


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    rho: float
    mae: float
    rmse: float
    bias: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "rho": self.rho,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "bias": self.bias,
        }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) <= 0 or np.std(right[valid]) <= 0:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def regression_metrics(
    observed: Sequence[float], predicted: Sequence[float]
) -> RegressionMetrics:
    """Return standard point-prediction diagnostics on common finite rows."""

    observation = np.asarray(observed, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    if observation.shape != prediction.shape or observation.ndim != 1:
        raise ValueError("observed and predicted must be aligned one-dimensional arrays")
    valid = np.isfinite(observation) & np.isfinite(prediction)
    if not valid.any():
        return RegressionMetrics(0, np.nan, np.nan, np.nan, np.nan)
    error = prediction[valid] - observation[valid]
    return RegressionMetrics(
        n=int(valid.sum()),
        rho=_safe_correlation(observation[valid], prediction[valid]),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(np.mean(error)),
    )


def _circular_block_indices(
    n: int, block_length: int, generator: np.random.Generator
) -> np.ndarray:
    block = max(1, min(int(block_length), n))
    starts = generator.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]


def moving_block_skill_interval(
    observed: Sequence[float],
    predicted: Sequence[float],
    baseline: Sequence[float],
    *,
    block_length: int,
    n_resamples: int = 999,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Estimate RMSE skill and a circular moving-block percentile interval.

    Skill is ``1 - RMSE(model) / RMSE(baseline)``.  Positive values improve on
    the baseline.  Resampling preserves local serial dependence up to the
    chosen block length in retained ROWS, not calendar duration. For gapped
    time series use calendar_statistics.calendar_skill_intervals instead.
    """

    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if obs.shape != pred.shape or obs.shape != base.shape or obs.ndim != 1:
        raise ValueError("observed, predicted, and baseline must be aligned vectors")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    valid = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(base)
    obs, pred, base = obs[valid], pred[valid], base[valid]
    if len(obs) < 10:
        return np.nan, np.nan, np.nan

    def score(index: np.ndarray) -> float:
        model_rmse = np.sqrt(np.mean((pred[index] - obs[index]) ** 2))
        baseline_rmse = np.sqrt(np.mean((base[index] - obs[index]) ** 2))
        return float(1.0 - model_rmse / baseline_rmse) if baseline_rmse > 0 else np.nan

    point = score(np.arange(len(obs)))
    generator = np.random.default_rng(seed)
    draws = np.asarray([
        score(_circular_block_indices(len(obs), block_length, generator))
        for _ in range(n_resamples)
    ])
    alpha = (1.0 - confidence) / 2.0
    return (
        point,
        float(np.nanquantile(draws, alpha)),
        float(np.nanquantile(draws, 1.0 - alpha)),
    )


def structured_period_shift_test(
    dates: Sequence[pd.Timestamp] | pd.Series,
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    ridge: float,
    sampling_interval_hours: int,
    period_hours: int = 168,
    n_permutations: int | None = None,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Test inverse-operator magnitude using whole-period circular shifts.

    The aligned series is first restored to its nominal calendar grid.  Output
    vectors are then shifted by whole periods, preserving hour-of-week when
    ``period_hours=168``. An incomplete final period is discarded before any
    operator is fitted or shifted. The returned tail rank has a test
    interpretation only under shift-invariance, not arbitrary nonstationarity.
    For explicit calendar anchoring and sample counts use
    calendar_statistics.complete_week_shift_diagnostic.
    """

    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    x = np.asarray(inputs, dtype=float)
    y = np.asarray(outputs, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) != len(date_index):
        raise ValueError("dates, inputs, and outputs must have aligned rows")
    if sampling_interval_hours < 1 or period_hours < sampling_interval_hours:
        raise ValueError("sampling and period hours are incompatible")
    if period_hours % sampling_interval_hours != 0:
        raise ValueError("period_hours must be divisible by sampling_interval_hours")
    if n_permutations is not None and n_permutations < 1:
        raise ValueError("n_permutations must be positive or None")
    grid = pd.date_range(
        date_index.min(), date_index.max(), freq=f"{sampling_interval_hours}h"
    )
    rows_per_period = period_hours // sampling_interval_hours
    # Circular wrapping must cover an integer number of periods. Otherwise
    # even a whole-period shift changes phase in its wrapped portion.
    complete_length = len(grid) // rows_per_period * rows_per_period
    grid = grid[:complete_length]
    x_frame = pd.DataFrame(x, index=date_index).reindex(grid)
    y_frame = pd.DataFrame(y, index=date_index).reindex(grid)
    observed_valid = x_frame.notna().all(axis=1) & y_frame.notna().all(axis=1)
    if observed_valid.sum() < 3:
        raise ValueError("too few complete rows for the observed inverse operator")
    observed = fit_inverse_operator(
        x_frame.loc[observed_valid].to_numpy(),
        y_frame.loc[observed_valid].to_numpy(),
        ridge=ridge,
    ).operator
    observed_norm = float(np.linalg.norm(observed))
    rows_per_period = period_hours // sampling_interval_hours
    maximum_cycles = len(grid) // rows_per_period
    if maximum_cycles < 2:
        raise ValueError("the series needs at least two complete null periods")
    admissible_cycles = np.arange(1, maximum_cycles, dtype=int)
    if n_permutations is None or n_permutations >= len(admissible_cycles):
        selected_cycles = admissible_cycles
    else:
        generator = np.random.default_rng(seed)
        selected_cycles = np.sort(
            generator.choice(admissible_cycles, size=n_permutations, replace=False)
        )
    y_values = y_frame.to_numpy()
    null_norms: list[float] = []
    for cycles in selected_cycles:
        shifted = np.roll(y_values, cycles * rows_per_period, axis=0)
        valid = x_frame.notna().all(axis=1).to_numpy() & np.isfinite(shifted).all(axis=1)
        if valid.sum() < 3:
            continue
        model = fit_inverse_operator(x_frame.to_numpy()[valid], shifted[valid], ridge=ridge)
        null_norms.append(float(np.linalg.norm(model.operator)))
    null = np.asarray(null_norms)
    p_value = (
        float((1 + np.sum(null >= observed_norm)) / (1 + len(null)))
        if len(null)
        else np.nan
    )
    return observed_norm, p_value, null
