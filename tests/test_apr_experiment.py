import numpy as np
import nbformat
import pandas as pd

from tsaime import ChronologicalSplit


NOTEBOOK = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "notebooks"
    / "tsAIME_APR_all_experiments_v9_package.ipynb"
)


def _preprocessing_namespace() -> dict[str, object]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(
        item
        for item in notebook.cells
        if "apr-preprocessing" in item.metadata.get("tags", [])
    )
    namespace: dict[str, object] = {"__name__": "apr_notebook_test"}
    exec(compile(cell.source, "apr-preprocessing", "exec"), namespace)
    return namespace


def _hourly_pm25_frame(periods: int = 24 * 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="h")
    values = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "PM25": values,
            "TEMP": 20.0 + values / 1000,
            "RH": 50.0,
            "PRESS": 1010.0,
            "RAIN": 0.0,
            "WIND_U": 1.0,
            "WIND_V": 0.0,
        }
    )


def test_future_pm25_targets_remain_exact_after_a_missing_hour() -> None:
    prepare_pm25_multihorizon_pairs = _preprocessing_namespace()[
        "prepare_pm25_multihorizon_pairs"
    ]
    frame = _hourly_pm25_frame()
    missing_time = pd.Timestamp("2024-01-05 06:00")
    frame = frame.loc[frame["Date"] != missing_time].copy()
    split = ChronologicalSplit(
        "2024-01-01", "2024-01-08", "2024-01-14", "2024-01-21"
    )
    _, complete, _, targets = prepare_pm25_multihorizon_pairs(
        frame,
        (1, 6, 24),
        split,
    )
    source = complete.loc[complete["Date"] == pd.Timestamp("2024-01-05 00:00")]
    assert source.empty
    unaffected = complete.loc[complete["Date"] == pd.Timestamp("2024-01-06 00:00")].iloc[0]
    assert unaffected[targets[1]] == frame.loc[
        frame["Date"] == pd.Timestamp("2024-01-06 06:00"), "PM25"
    ].iloc[0]


def test_meteorological_wind_from_direction_convention() -> None:
    wind_components = _preprocessing_namespace()["wind_components"]
    u, v = wind_components([2.0, 3.0], [90.0, 0.0])
    np.testing.assert_allclose(u, [-2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v, [0.0, -3.0], atol=1e-12)


def test_pm25_negative_policies_are_explicit_and_nonmutating() -> None:
    apply_pm25_policy = _preprocessing_namespace()["apply_pm25_policy"]
    source = pd.DataFrame({"PM25": [-2.0, 0.0, 5.0]})
    reported = apply_pm25_policy(source, "as_reported")
    missing = apply_pm25_policy(source, "negative_missing")
    zero = apply_pm25_policy(source, "negative_zero")
    assert reported["PM25"].tolist() == [-2.0, 0.0, 5.0]
    assert pd.isna(missing.loc[0, "PM25"])
    assert zero["PM25"].tolist() == [0.0, 0.0, 5.0]
    assert source["PM25"].tolist() == [-2.0, 0.0, 5.0]
