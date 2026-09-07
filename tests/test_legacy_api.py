import numpy as np
import pandas as pd

from tsaime import RollingTSAIME, RollingTSAIMEConfig


def test_v02_rolling_scalar_api_remains_available() -> None:
    generator = np.random.default_rng(11)
    n = 90
    x1 = generator.normal(size=n)
    x2 = generator.normal(size=n)
    frame = pd.DataFrame(
        {
            "Time": np.arange(1, n + 1),
            "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "Predictions": 0.8 * x1 - 0.2 * x2,
            "x1": x1,
            "x2": x2,
        }
    )
    result = RollingTSAIME(RollingTSAIMEConfig(window=21, step=3)).fit(
        frame, ["x1", "x2"]
    )
    assert list(result.global_scores.columns) == ["Time", "Date", "x1", "x2"]
    assert len(result.global_scores) == len(range(20, n, 3))

