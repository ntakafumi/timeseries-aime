import numpy as np
import pandas as pd

from tsaime import (
    ChronologicalSplit,
    QualityGate,
    audit_hourly_site,
    evaluate_quality_gate,
    prepare_chronological_pairs,
)


def make_hourly_frame(periods: int = 24 * 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="h")
    values = np.arange(periods, dtype=float)
    return pd.DataFrame({"Date": dates, "state": values, "target": values + 6.0})


def test_generic_pairs_keep_already_aligned_targets() -> None:
    frame = make_hourly_frame()
    missing_time = pd.Timestamp("2024-01-05 06:00")
    frame.loc[frame["Date"] == missing_time, "target"] = np.nan
    split = ChronologicalSplit(
        "2024-01-01", "2024-01-08", "2024-01-14", "2024-01-21"
    )
    _, complete = prepare_chronological_pairs(
        frame,
        ["state"],
        ["target"],
        split,
        max_target_horizon_hours=6,
    )
    assert missing_time not in set(complete["Date"])
    unaffected = complete.loc[complete["Date"] == pd.Timestamp("2024-01-06 00:00")].iloc[0]
    assert unaffected["target"] == unaffected["state"] + 6.0


def test_target_cannot_cross_a_split_boundary() -> None:
    frame = make_hourly_frame()
    split = ChronologicalSplit(
        "2024-01-01", "2024-01-08", "2024-01-14", "2024-01-21"
    )
    candidates, _ = prepare_chronological_pairs(
        frame,
        ["state"],
        ["target"],
        split,
        max_target_horizon_hours=24,
    )
    assert pd.Timestamp("2024-01-07 00:00") not in set(candidates["Date"])


def test_quality_gate_reports_each_split() -> None:
    candidates = pd.DataFrame({"split": ["train"] * 10 + ["validation"] * 5 + ["test"] * 5})
    complete = candidates.iloc[:-1]
    gate = QualityGate({"train": 8, "validation": 4, "test": 5}, minimum_coverage=0.8)
    report = evaluate_quality_gate(candidates, complete, gate)
    assert report["quality_gate"] == "FAIL"
    assert "test" in str(report["reason"])


def test_audit_uses_half_open_study_bounds() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2022-12-31 22:00", periods=4, freq="h"),
            "value": [1.0, 1.0, 1.0, 1.0],
        }
    )
    audit = audit_hourly_site(
        frame,
        ["value"],
        start="2022-12-31 22:00",
        end="2023-01-01",
    )
    assert audit["year"].tolist() == [2022]
    assert audit["expected_hours"].tolist() == [2]


def test_audit_pads_missing_study_period_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01 01:00", periods=8758, freq="h"),
            "value": np.ones(8758),
        }
    )
    audit = audit_hourly_site(
        frame,
        ["value"],
        start="2024-01-01",
        end="2025-01-01",
    )
    assert audit["expected_hours"].tolist() == [8784]
    assert audit["value_max_gap_h"].tolist() == [25]
    assert audit["value_coverage"].iloc[0] == 8758 / 8784
