"""Calendar-aware paired loss resampling and phase-preserving shift diagnostics.

These functions contain no dataset names or application-specific variables.
Intervals condition on supplied fitted predictions; they do not refit a model.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from .operators import fit_inverse_operator


def calendar_frame(dates: Sequence, values: np.ndarray, interval_hours: int) -> pd.DataFrame:
    """Restore missing grid positions; reject duplicate/off-grid timestamps."""
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if interval_hours <= 0 or len(dates) != len(values) or len(dates) < 2:
        raise ValueError("invalid dates, values, or interval")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("dates must be unique, finite, and sorted")
    step = pd.Timedelta(hours=interval_hours).value
    # asi8 follows the index storage unit (us in pandas 3); Timedelta.value
    # is ns. Normalize the calendar check without changing timestamps/losses.
    dates_ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if np.any((dates_ns - dates_ns[0]) % step):
        raise ValueError("timestamps are not on the declared calendar grid")
    grid = pd.date_range(dates[0], dates[-1], freq=pd.Timedelta(hours=interval_hours))
    return pd.DataFrame(values, index=dates).reindex(grid)


def calendar_skill_intervals(
    dates: Sequence, observed: np.ndarray, predicted: np.ndarray, baseline: np.ndarray,
    *, interval_hours: int, block_days: Sequence[int] = (7, 14, 28),
    n_resamples: int = 1999, seed: int = 0, confidence: float = .95,
) -> pd.DataFrame:
    """Paired RMSE-skill percentile intervals using non-circular calendar blocks.

    Columns are comparisons. Missing positions remain in sampled blocks.
    Cumulative squared losses avoid materializing B copies of the full data.
    Each resample contains n grid positions, including a final partial block.
    The same block draws are used for every comparison column.
    """
    arrays = [np.asarray(a, dtype=float) for a in (observed, predicted, baseline)]
    arrays = [a[:, None] if a.ndim == 1 else a for a in arrays]
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("all arrays must have identical n-by-k shapes")
    if n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid resampling parameters")
    k = arrays[0].shape[1]
    frame = calendar_frame(dates, np.concatenate(arrays, axis=1), interval_hours)
    obs, pred, base = np.split(frame.to_numpy(), [k, 2*k], axis=1)
    valid = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(base)
    loss = np.where(valid, (pred-obs)**2, 0.)
    base_loss = np.where(valid, (base-obs)**2, 0.)
    joint = np.concatenate([loss, base_loss], axis=1)
    prefix = np.vstack([np.zeros((1, 2*k)), np.cumsum(joint, axis=0)])
    n = len(frame)
    count = valid.sum(axis=0)
    total = joint.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        point = 1-np.sqrt(total[:k]/total[k:])
    rows = []
    for days in block_days:
        if days <= 0 or (24*days) % interval_hours:
            raise ValueError("block duration must be positive and grid aligned")
        length = int(24*days/interval_hours)
        if length >= n:
            raise ValueError("block length must be shorter than the observed calendar span")
        rng = np.random.default_rng(seed)
        full, remainder = divmod(n, length)
        sums = np.zeros((n_resamples, 2*k))
        for width in [length]*full + ([remainder] if remainder else []):
            starts = rng.integers(0, n-width+1, n_resamples)
            sums += prefix[starts+width]-prefix[starts]
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = 1-np.sqrt(sums[:, :k]/sums[:, k:])
        for j in range(k):
            finite = draws[:, j][np.isfinite(draws[:, j])]
            enough = count[j] >= 10 and len(finite) >= .95*n_resamples and total[k+j] > 0
            low, high = np.quantile(finite, [(1-confidence)/2, (1+confidence)/2]) if enough else (np.nan, np.nan)
            rows.append(dict(comparison_index=j, block_days=int(days), n=int(count[j]),
                             grid_n=n, skill=float(point[j]) if enough else np.nan,
                             CI_low=float(low), CI_high=float(high),
                             bootstrap_valid_draws=len(finite),
                             method="noncircular_calendar_moving_block_percentile",
                             uncertainty_scope="pointwise_conditional_on_fitted_predictions"))
    return pd.DataFrame(rows)


def complete_week_shift_diagnostic(
    dates: Sequence, inputs: np.ndarray, outputs: np.ndarray, *, ridge: float,
    interval_hours: int, n_shifts: int | None = None, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare operator norm to whole-week shifts on complete Monday-based weeks.

    The tail rank assumes shift-invariance to have a test interpretation.
    Seasonal nonstationarity is not removed. Never call this an exact p-value
    for arbitrary environmental time series. Per-shift valid counts are saved.
    """
    x, y = np.asarray(inputs, float), np.asarray(outputs, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or interval_hours <= 0 or 24 % interval_hours:
        raise ValueError("invalid dimensions or interval")
    d = x.shape[1]
    frame = calendar_frame(dates, np.column_stack([x, y]), interval_hours)
    beginning = frame.index[0].normalize()
    beginning += pd.Timedelta(days=(7-beginning.weekday()) % 7)
    if beginning < frame.index[0]:
        beginning += pd.Timedelta(days=7)
    ending = frame.index[-1] + pd.Timedelta(hours=interval_hours)
    weeks = int((ending-beginning)//pd.Timedelta(days=7))
    if weeks < 3:
        raise ValueError("at least three full calendar weeks are required")
    grid = pd.date_range(beginning, periods=weeks*168//interval_hours,
                         freq=pd.Timedelta(hours=interval_hours))
    if not grid.isin(frame.index).all():
        raise ValueError("grid must align to Monday midnight")
    kept = frame.reindex(grid).to_numpy()
    xx, yy = kept[:, :d], kept[:, d:]
    mask = np.isfinite(kept).all(axis=1)
    observed = np.linalg.norm(fit_inverse_operator(xx[mask], yy[mask], ridge=ridge).operator)
    shifts = np.arange(1, weeks)
    if n_shifts is not None:
        if n_shifts < 1:
            raise ValueError("n_shifts must be positive")
        if n_shifts < len(shifts):
            shifts = np.sort(np.random.default_rng(seed).choice(shifts, n_shifts, replace=False))
    rows = []
    phase = grid.weekday.to_numpy()*24 + grid.hour.to_numpy()
    for shift in shifts:
        rolled = np.roll(yy, int(shift)*168//interval_hours, axis=0)
        mismatch = int(np.sum(phase != np.roll(phase, int(shift)*168//interval_hours)))
        if mismatch:
            raise RuntimeError("internal calendar phase invariant failed")
        finite = np.isfinite(xx).all(axis=1) & np.isfinite(rolled).all(axis=1)
        if finite.sum() < 3:
            raise ValueError("a shift has too few paired observations")
        norm = np.linalg.norm(fit_inverse_operator(xx[finite], rolled[finite], ridge=ridge).operator)
        rows.append(dict(shift_weeks=int(shift), paired_n=int(finite.sum()), operator_norm=float(norm),
                         phase_mismatch_count=mismatch))
    detail = pd.DataFrame(rows)
    tail_rank = (1+np.sum(detail.operator_norm >= observed))/(1+len(detail))
    summary = pd.DataFrame([dict(observed_operator_norm=float(observed), shift_tail_rank=float(tail_rank),
        observed_n=int(mask.sum()), full_weeks=weeks, grid_start=grid[0],
        grid_end_exclusive=grid[-1]+pd.Timedelta(hours=interval_hours),
        discarded_grid_rows=len(frame)-len(grid), null_unique_shifts=len(detail),
        minimum_attainable_rank=1/(1+len(detail)), null_median=float(detail.operator_norm.median()),
        null_95pct=float(detail.operator_norm.quantile(.95)), phase_mismatch_count=int(detail.phase_mismatch_count.max()),
        interpretation="shift-invariance diagnostic; not exact under seasonal nonstationarity")])
    return summary, detail
