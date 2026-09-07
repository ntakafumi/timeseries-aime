"""Time-windowed vector-output ts-AIME estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .operators import fit_inverse_operator


@dataclass(frozen=True)
class RollingVectorTSAIMEConfig:
    """Configuration for exact-calendar rolling inverse operators."""

    window: str | pd.Timedelta = "28D"
    step: str | pd.Timedelta = "7D"
    ridge: float = 0.01
    min_samples: int = 30

    def validate(self) -> None:
        window = pd.Timedelta(self.window)
        step = pd.Timedelta(self.step)
        if window <= pd.Timedelta(0):
            raise ValueError("window must be positive")
        if step <= pd.Timedelta(0):
            raise ValueError("step must be positive")
        if self.ridge < 0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")
        if self.min_samples < 3:
            raise ValueError("min_samples must be at least 3")


@dataclass(frozen=True)
class RollingVectorTSAIMEResult:
    """Tidy operator and diagnostic tables from a rolling fit."""

    operators: pd.DataFrame
    diagnostics: pd.DataFrame
    features: tuple[str, ...]
    outputs: tuple[str, ...]
    config: RollingVectorTSAIMEConfig

    def metadata(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "outputs": list(self.outputs),
            "config": {
                **asdict(self.config),
                "window": str(pd.Timedelta(self.config.window)),
                "step": str(pd.Timedelta(self.config.step)),
            },
            "n_windows": int(len(self.diagnostics)),
            "n_operator_rows": int(len(self.operators)),
        }


class RollingVectorTSAIME:
    """Estimate vector-output ts-AIME on trailing calendar windows."""

    def __init__(self, config: RollingVectorTSAIMEConfig):
        config.validate()
        self.config = config

    def fit(
        self,
        aligned: pd.DataFrame,
        features: Sequence[str],
        outputs: Sequence[str],
        *,
        date_column: str = "Date",
    ) -> RollingVectorTSAIMEResult:
        """Fit operators to an already aligned source-time forecast table.

        The method never shifts rows.  ``aligned`` must already pair each state
        at source time with its forecast vector, and ``date_column`` must retain
        the actual source timestamp.
        """

        feature_names = tuple(str(value) for value in features)
        output_names = tuple(str(value) for value in outputs)
        if not feature_names or not output_names:
            raise ValueError("features and outputs must both be non-empty")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("features must not contain duplicates")
        if len(set(output_names)) != len(output_names):
            raise ValueError("outputs must not contain duplicates")
        required = [date_column, *feature_names, *output_names]
        missing = [column for column in required if column not in aligned]
        if missing:
            raise KeyError(f"aligned data is missing required columns: {missing}")

        data = aligned[required].copy()
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        if data[date_column].isna().any():
            raise ValueError("date_column contains invalid timestamps")
        if data[date_column].duplicated().any():
            raise ValueError("date_column must contain unique timestamps")
        data = data.sort_values(date_column).reset_index(drop=True)

        window = pd.Timedelta(self.config.window)
        step = pd.Timedelta(self.config.step)
        endpoint = data[date_column].min() + window
        last_date = data[date_column].max()
        operator_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []

        while endpoint <= last_date:
            start = endpoint - window
            selected = data.loc[
                (data[date_column] > start) & (data[date_column] <= endpoint)
            ].dropna(subset=[*feature_names, *output_names])
            if len(selected) >= self.config.min_samples:
                model = fit_inverse_operator(
                    selected[list(feature_names)].to_numpy(float),
                    selected[list(output_names)].to_numpy(float),
                    ridge=self.config.ridge,
                )
                condition = float(np.linalg.cond(
                    model.output_covariance
                    + self.config.ridge * np.eye(len(output_names))
                ))
                diagnostic_rows.append(
                    {
                        "endpoint": endpoint,
                        "window_start_exclusive": start,
                        "n_window": int(len(selected)),
                        "ridge": self.config.ridge,
                        "operator_frobenius_norm": float(np.linalg.norm(model.operator)),
                        "output_covariance_condition": condition,
                    }
                )
                for feature_index, feature in enumerate(feature_names):
                    for output_index, output in enumerate(output_names):
                        operator_rows.append(
                            {
                                "endpoint": endpoint,
                                "feature": feature,
                                "output": output,
                                "coefficient": float(model.operator[feature_index, output_index]),
                            }
                        )
            endpoint += step

        return RollingVectorTSAIMEResult(
            operators=pd.DataFrame(operator_rows),
            diagnostics=pd.DataFrame(diagnostic_rows),
            features=feature_names,
            outputs=output_names,
            config=self.config,
        )

