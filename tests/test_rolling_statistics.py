import numpy as np
import pandas as pd

from tsaime import (
    RollingVectorTSAIME,
    RollingVectorTSAIMEConfig,
    moving_block_skill_interval,
    structured_period_shift_test,
)


def test_rolling_vector_uses_calendar_windows() -> None:
    generator = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=24 * 35, freq="h")
    x = generator.normal(size=(len(dates), 2))
    y = np.column_stack([x[:, 0] + 0.1 * x[:, 1], x[:, 1] - 0.2 * x[:, 0]])
    frame = pd.DataFrame(
        {"Date": dates, "x1": x[:, 0], "x2": x[:, 1], "y1": y[:, 0], "y2": y[:, 1]}
    )
    fitted = RollingVectorTSAIME(
        RollingVectorTSAIMEConfig(window="14D", step="7D", ridge=0.01, min_samples=100)
    ).fit(frame, ["x1", "x2"], ["y1", "y2"])
    assert len(fitted.diagnostics) == 3
    assert set(fitted.operators["feature"]) == {"x1", "x2"}
    assert set(fitted.operators["output"]) == {"y1", "y2"}


def test_moving_block_interval_is_reproducible_and_positive_for_better_model() -> None:
    generator = np.random.default_rng(2)
    observed = generator.normal(size=600)
    prediction = observed + generator.normal(scale=0.2, size=600)
    baseline = observed + generator.normal(scale=1.0, size=600)
    first = moving_block_skill_interval(
        observed, prediction, baseline, block_length=24, n_resamples=199, seed=5
    )
    second = moving_block_skill_interval(
        observed, prediction, baseline, block_length=24, n_resamples=199, seed=5
    )
    np.testing.assert_allclose(first, second)
    assert first[0] > 0 and first[1] > 0


def test_structured_weekly_null_is_reproducible() -> None:
    generator = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=24 * 70, freq="h")
    inputs = generator.normal(size=(len(dates), 3))
    outputs = np.column_stack([inputs[:, 0], inputs[:, 1]]) + generator.normal(
        scale=0.1, size=(len(dates), 2)
    )
    first = structured_period_shift_test(
        dates,
        inputs,
        outputs,
        ridge=0.01,
        sampling_interval_hours=1,
        n_permutations=29,
        seed=7,
    )
    second = structured_period_shift_test(
        dates,
        inputs,
        outputs,
        ridge=0.01,
        sampling_interval_hours=1,
        n_permutations=29,
        seed=7,
    )
    assert first[1] == second[1]
    np.testing.assert_allclose(first[2], second[2])
    assert len(first[2]) == 9


def test_structured_weekly_null_enumerates_each_admissible_shift_once() -> None:
    generator = np.random.default_rng(4)
    dates = pd.date_range("2024-01-01", periods=24 * 35, freq="h")
    inputs = generator.normal(size=(len(dates), 2))
    outputs = inputs[:, :1] + generator.normal(scale=0.2, size=(len(dates), 1))
    observed, p_value, null = structured_period_shift_test(
        dates,
        inputs,
        outputs,
        ridge=0.01,
        sampling_interval_hours=1,
        n_permutations=None,
    )
    assert np.isfinite(observed)
    assert len(null) == 4
    assert p_value >= 1 / 5
