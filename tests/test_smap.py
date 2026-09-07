import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyEDM")

from tsaime.smap import run_smap_oos, select_smap_theta


def make_table(n: int = 240) -> pd.DataFrame:
    generator = np.random.default_rng(10)
    x = generator.normal(size=(n, 3))
    y = np.column_stack([x[:, 0] + 0.2 * x[:, 1], x[:, 2] - 0.1 * x[:, 0]])
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=n, freq="h"),
            "x1": x[:, 0],
            "x2": x[:, 1],
            "x3": x[:, 2],
            "y1": y[:, 0],
            "y2": y[:, 1],
        }
    )


def test_smap_is_strictly_out_of_sample_and_returns_forward_shape() -> None:
    table = make_table()
    result = run_smap_oos(
        table.iloc[:160], table.iloc[160:], ["x1", "x2", "x3"], ["y1", "y2"], theta=0.0
    )
    assert result.predictions.shape == (80, 2)
    assert result.coefficients_standardized.shape == (80, 2, 3)
    assert result.dates.min() > table.iloc[:160]["Date"].max()


def test_smap_rejects_overlapping_library_and_prediction() -> None:
    table = make_table()
    with pytest.raises(ValueError, match="strictly after"):
        run_smap_oos(
            table.iloc[:160], table.iloc[150:], ["x1", "x2", "x3"], ["y1", "y2"], theta=0.0
        )


def test_theta_selection_uses_validation_only() -> None:
    table = make_table()
    theta, scores = select_smap_theta(
        table.iloc[:160],
        table.iloc[160:],
        ["x1", "x2", "x3"],
        ["y1", "y2"],
        [0.0, 2.0],
        validation_stride=2,
    )
    assert theta in {0.0, 2.0}
    assert len(scores) == 2

