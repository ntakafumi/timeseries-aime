"""Core scalar and vector-output ts-AIME operators.

The module contains no forecasting code.  It receives source-time states and
already aligned forecast outputs, keeping the forecast model and the inverse
explanation operator as separate estimands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _as_matrix(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three rows")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class MatrixStandardizer:
    """Column-wise population standardization parameters."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: Sequence[Sequence[float]] | np.ndarray) -> "MatrixStandardizer":
        array = _as_matrix(values, "values")
        mean = np.mean(array, axis=0)
        scale = np.std(array, axis=0, ddof=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(self.mean):
            raise ValueError("values have an incompatible shape")
        return (array - self.mean) / self.scale

    def inverse_transform(self, values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(self.mean):
            raise ValueError("values have an incompatible shape")
        return array * self.scale + self.mean


@dataclass(frozen=True)
class InverseOperatorModel:
    """Fitted standardized output-to-input reconstruction operator.

    ``operator`` has shape ``(n_features, n_outputs)`` and implements
    ``X_hat_z = Y_z @ operator.T``.
    """

    operator: np.ndarray
    cross_covariance: np.ndarray
    output_covariance: np.ndarray
    x_scaler: MatrixStandardizer
    y_scaler: MatrixStandardizer
    ridge: float

    def reconstruct(self, outputs: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Reconstruct inputs in their original units from output vectors."""

        output_z = self.y_scaler.transform(outputs)
        input_z = output_z @ self.operator.T
        return self.x_scaler.inverse_transform(input_z)

    def reconstruct_standardized(
        self, outputs: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        """Reconstruct inputs in the fitted standardized input coordinates."""

        return self.y_scaler.transform(outputs) @ self.operator.T


def fit_inverse_operator(
    inputs: Sequence[Sequence[float]] | np.ndarray,
    outputs: Sequence[Sequence[float]] | np.ndarray,
    *,
    ridge: float = 0.0,
) -> InverseOperatorModel:
    r"""Fit the regularized vector-output ts-AIME operator.

    For column-standardized matrices ``X`` and ``Y`` the estimate is

    ``A = S_XY @ pinv(S_YY + ridge * I)``.

    The returned operator maps standardized outputs to standardized inputs.
    """

    x = _as_matrix(inputs, "inputs")
    y = _as_matrix(outputs, "outputs")
    if len(x) != len(y):
        raise ValueError("inputs and outputs must have the same number of rows")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be a finite non-negative value")
    x_scaler = MatrixStandardizer.fit(x)
    y_scaler = MatrixStandardizer.fit(y)
    x_z = x_scaler.transform(x)
    y_z = y_scaler.transform(y)
    n = len(x_z)
    s_xy = x_z.T @ y_z / n
    s_yy = y_z.T @ y_z / n
    operator = s_xy @ np.linalg.pinv(s_yy + float(ridge) * np.eye(y_z.shape[1]))
    return InverseOperatorModel(
        operator=operator,
        cross_covariance=s_xy,
        output_covariance=s_yy,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        ridge=float(ridge),
    )


def scalar_aime_operator(
    inputs: Sequence[Sequence[float]] | np.ndarray,
    output: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the scalar-output standardized operator.

    For finite nonconstant columns this is exactly the vector of Pearson
    correlations.  The second result is the endpoint Hadamard contribution
    used by the v0.2 scalar API.  Constant input columns are returned as
    ``NaN`` in both arrays.
    """

    x = np.asarray(inputs, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("inputs must be a two-dimensional matrix with at least three rows")
    y = np.asarray(output, dtype=float)
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("output must be one-dimensional and aligned with inputs")
    if not np.isfinite(y).all():
        missing = np.full(x.shape[1], np.nan)
        return missing, missing.copy()
    y_sd = float(np.std(y, ddof=0))
    if y_sd <= 1e-12:
        missing = np.full(x.shape[1], np.nan)
        return missing, missing.copy()
    y_z = (y - np.mean(y)) / y_sd
    result = np.full(x.shape[1], np.nan)
    local = np.full(x.shape[1], np.nan)
    for column in range(x.shape[1]):
        if not np.isfinite(x[:, column]).all():
            continue
        x_sd = float(np.std(x[:, column], ddof=0))
        if x_sd > 1e-12:
            x_z = (x[:, column] - np.mean(x[:, column])) / x_sd
            result[column] = float(x_z @ y_z / len(x_z))
            local[column] = float(y_z[-1] * result[column] * x_z[-1])
    return result, local


def marginal_cross_correlation(
    inputs: Sequence[Sequence[float]] | np.ndarray,
    outputs: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return standardized cross-covariances without ``S_YY^{-1}``.

    This matrix is a collection of marginal correlations and is a baseline,
    not the vector-output inverse operator when outputs are dependent.
    """

    return fit_inverse_operator(inputs, outputs).cross_covariance


def covariance_weighted_inverse(
    forward: Sequence[Sequence[float]] | np.ndarray,
    state_covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    ridge: float = 0.0,
    output_noise_covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the inverse implied by a forward operator and state covariance.

    ``forward`` has shape ``(n_outputs, n_features)`` and the result has shape
    ``(n_features, n_outputs)``:

    ``A = Sigma @ F.T @ pinv(F @ Sigma @ F.T + Sigma_eps + ridge * I)``.

    The expression assumes zero covariance between the state and additive
    residual/noise. Arbitrary local-fit residuals need not satisfy this.
    ``output_noise_covariance`` must use the same output coordinates as
    ``forward``. Omitting it selects the noiseless special case.
    """

    f = np.asarray(forward, dtype=float)
    sigma = np.asarray(state_covariance, dtype=float)
    if f.ndim != 2:
        raise ValueError("forward must be two-dimensional")
    if sigma.shape != (f.shape[1], f.shape[1]):
        raise ValueError("state_covariance has an incompatible shape")
    if not np.isfinite(f).all() or not np.isfinite(sigma).all():
        raise ValueError("forward and state_covariance must be finite")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be a finite non-negative value")
    if output_noise_covariance is None:
        noise = np.zeros((f.shape[0], f.shape[0]), dtype=float)
    else:
        noise = np.asarray(output_noise_covariance, dtype=float)
        if noise.shape != (f.shape[0], f.shape[0]):
            raise ValueError("output_noise_covariance has an incompatible shape")
        if not np.isfinite(noise).all() or not np.allclose(noise, noise.T, atol=1e-10):
            raise ValueError("output_noise_covariance must be finite and symmetric")
        if np.linalg.eigvalsh(noise).min() < -1e-10:
            raise ValueError("output_noise_covariance must be positive semidefinite")
    return sigma @ f.T @ np.linalg.pinv(
        f @ sigma @ f.T + noise + float(ridge) * np.eye(f.shape[0])
    )


def operator_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity after vectorizing two equal-shaped operators."""

    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("operators must have the same shape")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.ravel() @ b.ravel() / denominator) if denominator > 0 else np.nan
