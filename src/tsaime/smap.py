"""Strictly out-of-sample S-Map utilities for explicit future-target columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .operators import MatrixStandardizer


@dataclass(frozen=True)
class SMapForecast:
    """Predictions and local forward coefficient matrices from one S-Map run."""

    predictions: np.ndarray
    observations: np.ndarray
    predictions_standardized: np.ndarray
    observations_standardized: np.ndarray
    coefficients_standardized: np.ndarray
    dates: pd.Series
    input_scaler: MatrixStandardizer
    output_scaler: MatrixStandardizer
    theta: float


def _import_pyedm():
    try:
        import pyEDM
    except ImportError as exc:
        raise ImportError(
            "S-Map support requires the experiment extra: "
            "python -m pip install 'tsaime[smap]'"
        ) from exc
    return pyEDM


def _validate_columns(
    frame: pd.DataFrame, features: Sequence[str], targets: Sequence[str], date_column: str
) -> None:
    required = [date_column, *features, *targets]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"missing S-Map columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("S-Map library and prediction rows must be complete")


def run_smap_oos(
    library: pd.DataFrame,
    prediction: pd.DataFrame,
    features: Sequence[str],
    targets: Sequence[str],
    *,
    theta: float,
    date_column: str = "Date",
) -> SMapForecast:
    """Fit on ``library`` and predict only the later ``prediction`` rows.

    Targets must already be aligned to their source-time states.  The function
    therefore calls S-Map with ``Tp=0`` and ``embedded=True``.  This avoids any
    implicit row shift after missing timestamps have been removed.
    """

    pyedm = _import_pyedm()
    feature_names = tuple(str(value) for value in features)
    target_names = tuple(str(value) for value in targets)
    if not feature_names or not target_names:
        raise ValueError("features and targets must both be non-empty")
    _validate_columns(library, feature_names, target_names, date_column)
    _validate_columns(prediction, feature_names, target_names, date_column)
    library_dates = pd.to_datetime(library[date_column])
    prediction_dates = pd.to_datetime(prediction[date_column])
    if library_dates.max() >= prediction_dates.min():
        raise ValueError("prediction rows must occur strictly after all library rows")
    if theta < 0 or not np.isfinite(theta):
        raise ValueError("theta must be finite and non-negative")

    input_scaler = MatrixStandardizer.fit(library[list(feature_names)].to_numpy(float))
    output_scaler = MatrixStandardizer.fit(library[list(target_names)].to_numpy(float))
    parts: list[pd.DataFrame] = []
    for frame in (library, prediction):
        part = pd.DataFrame(
            input_scaler.transform(frame[list(feature_names)].to_numpy(float)),
            columns=feature_names,
        )
        target_z = output_scaler.transform(frame[list(target_names)].to_numpy(float))
        for index, target in enumerate(target_names):
            part[target] = target_z[:, index]
        part[date_column] = pd.to_datetime(frame[date_column]).to_numpy()
        parts.append(part)
    model = pd.concat(parts, ignore_index=True)
    model.insert(0, "Time", np.arange(1, len(model) + 1))
    library_n = len(library)
    predictions_z: list[np.ndarray] = []
    observations_z: list[np.ndarray] = []
    coefficient_matrices: list[np.ndarray] = []
    for target in target_names:
        result = pyedm.SMap(
            dataFrame=model[["Time", *feature_names, target]],
            columns=" ".join(feature_names),
            target=target,
            lib=f"1 {library_n}",
            pred=f"{library_n + 1} {len(model)}",
            E=len(feature_names),
            embedded=True,
            theta=float(theta),
            Tp=0,
            showPlot=False,
            verbose=False,
        )
        prediction_table = result["predictions"].reset_index(drop=True)
        coefficient_table = result["coefficients"].reset_index(drop=True)
        predictions_z.append(
            pd.to_numeric(prediction_table["Predictions"], errors="coerce").to_numpy()
        )
        observations_z.append(
            pd.to_numeric(prediction_table["Observations"], errors="coerce").to_numpy()
        )
        coefficients = coefficient_table.iloc[:, -len(feature_names):].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy()
        coefficient_matrices.append(coefficients)
    prediction_z = np.column_stack(predictions_z)
    observation_z = np.column_stack(observations_z)
    if not np.isfinite(prediction_z).all():
        raise RuntimeError("S-Map returned non-finite out-of-sample predictions")
    return SMapForecast(
        predictions=output_scaler.inverse_transform(prediction_z),
        observations=output_scaler.inverse_transform(observation_z),
        predictions_standardized=prediction_z,
        observations_standardized=observation_z,
        coefficients_standardized=np.stack(coefficient_matrices, axis=1),
        dates=prediction[date_column].reset_index(drop=True),
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        theta=float(theta),
    )


def select_smap_theta(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    targets: Sequence[str],
    theta_grid: Sequence[float],
    *,
    validation_stride: int = 1,
    date_column: str = "Date",
) -> tuple[float, pd.DataFrame]:
    """Select localization on validation data without access to test rows."""

    if validation_stride < 1:
        raise ValueError("validation_stride must be positive")
    thinned = validation.iloc[::validation_stride].reset_index(drop=True)
    rows = []
    for theta in theta_grid:
        result = run_smap_oos(
            train,
            thinned,
            features,
            targets,
            theta=float(theta),
            date_column=date_column,
        )
        rmse = float(np.sqrt(np.mean(
            (result.predictions_standardized - result.observations_standardized) ** 2
        )))
        rows.append({"theta": float(theta), "validation_standardized_RMSE": rmse})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("theta_grid must be non-empty")
    best = table.sort_values(["validation_standardized_RMSE", "theta"]).iloc[0]
    return float(best["theta"]), table


def coefficients_in_window_coordinates(
    coefficients_standardized: np.ndarray,
    input_window_standardized: np.ndarray,
    output_window_standardized: np.ndarray,
) -> np.ndarray:
    """Convert globally standardized S-Map coefficients to window coordinates."""

    coefficients = np.asarray(coefficients_standardized, dtype=float)
    x = np.asarray(input_window_standardized, dtype=float)
    y = np.asarray(output_window_standardized, dtype=float)
    if coefficients.ndim != 3 or x.ndim != 2 or y.ndim != 2:
        raise ValueError("coefficients, inputs, and outputs have incompatible ranks")
    if len(coefficients) != len(x) or len(x) != len(y):
        raise ValueError("coefficients, inputs, and outputs must have aligned rows")
    if coefficients.shape[1:] != (y.shape[1], x.shape[1]):
        raise ValueError("coefficient matrix dimensions do not match inputs and outputs")
    x_scale = np.std(x, axis=0, ddof=0)
    y_scale = np.std(y, axis=0, ddof=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    y_scale = np.where(y_scale > 1e-12, y_scale, 1.0)
    return coefficients * x_scale[None, None, :] / y_scale[None, :, None]
