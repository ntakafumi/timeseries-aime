"""Notebook-owned APR explanation diagnostics; no separate experiment package."""
from pathlib import Path
import hashlib
import json

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "tsAIME_APR_all_experiments_v9_4.ipynb"


@pytest.fixture(scope="module")
def ns():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    scope = {"PACKAGE_ROOT": ROOT}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and (index in (5, 7, 9, 11, 13, 15) or
                                         "v94-definitions" in cell.metadata.get("tags", [])):
            exec(compile(cell.source, f"v94-cell-{index}", "exec"), scope)
    return scope


def test_notebook_is_clean_compilable_and_contains_the_whole_workflow():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{index}", "exec")
            assert cell.execution_count is None and not cell.outputs
    source = "\n".join(c.source for c in notebook.cells)
    assert "from experiments" not in source
    assert "/private/tmp/" not in source and "/Users/" not in source
    for function in ("_run_site_v93", "_synthetic_v93", "_explain_site_v94", "_foundation_v94", "run_apr_pm25"):
        assert f"def {function}(" in source
    assert '"auto"' in source and '"publication"' in source


def test_bootstrap_weights_preserve_calendar_gaps(ns):
    dates = pd.date_range("2024-01-01", periods=112, freq="6h").delete([11, 12, 45, 78])
    weights, grid_n = ns["_calendar_resample_weights"](dates, 6, 7, 49, 11)
    assert grid_n == 112 and weights.shape == (49, 108)
    assert np.all(weights >= 0) and (weights.sum(axis=1) < 112).any()
    rng = np.random.default_rng(11)
    lookup = np.full(112, -1)
    lookup[pd.date_range(dates[0], dates[-1], freq="6h").get_indexer(dates)] = np.arange(108)
    expected = np.zeros_like(weights)
    for _ in range(4):
        starts = rng.integers(0, 85, size=49)
        for k, start in enumerate(starts):
            picked = lookup[start:start+28]
            np.add.at(expected[k], picked[picked >= 0], 1)
    np.testing.assert_array_equal(weights, expected)


def test_batched_resampling_exactly_matches_generic_inverse_fit(ns):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(112, 10))
    x[:, 5] = 1010 + x[:, 5] * 4
    x[:, 6] = 0  # A constant rain column must not be promoted to stable importance.
    y = x[:, :3] @ rng.normal(size=(3, 3)) + rng.normal(size=(112, 3)) * .2
    w, _ = ns["_calendar_resample_weights"](pd.date_range("2024-01-01", periods=112, freq="6h"), 6, 7, 19, 12)
    batch = ns["_weighted_inverse_batch"](x, y, w, .01)
    for k in (0, 3, 11):
        idx = np.repeat(np.arange(len(x)), w[k].astype(int))
        reference = ns["fit_inverse_operator"](x[idx], y[idx], ridge=.01)
        np.testing.assert_allclose(batch["operator"][k], reference.operator, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(batch["x_mean"][k], reference.x_scaler.mean, atol=1e-10)
        np.testing.assert_allclose(batch["y_mean"][k], reference.y_scaler.mean, atol=1e-10)
        assert not batch["input_active"][k, 6]


def test_stability_is_invariant_to_physical_unit_rescaling(ns):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(112, 10)); y = x[:, :3] + rng.normal(size=(112, 3)) * .2
    new_y = y[:10] + .2
    dates = pd.date_range("2024-01-01", periods=112, freq="6h")
    config = ns["APRExperimentConfig"](run_mode="smoke", origin_step_hours=6).resolved()
    e = ns["ExplanationConfig"](stability_resamples=19, stability_block_days=(7,))
    model = ns["fit_inverse_operator"](x, y, ridge=.01)
    first, f1 = ns["_stability_window"](x, y, new_y, dates, model, config, e, 98)
    scale = np.arange(1, 11) * 100.
    second_model = ns["fit_inverse_operator"](x * scale, y, ridge=.01)
    second, f2 = ns["_stability_window"](x * scale, y, new_y, dates, second_model, config, e, 98)
    for key in ("coefficient", "resample_q025", "resample_q975", "sign_agreement"):
        np.testing.assert_allclose(pd.DataFrame(first)[key], pd.DataFrame(second)[key], atol=1e-9)
    np.testing.assert_allclose(pd.DataFrame(f1).reconstruction_drift_median,
                               pd.DataFrame(f2).reconstruction_drift_median, atol=1e-9)


def test_event_thresholds_ignore_test_values_and_preserve_quantile_semantics(ns):
    rng = np.random.default_rng(31)
    vx = rng.normal(size=(200, 10)); vy = rng.normal(size=(200, 3))
    vy[:, 2] += 10  # Even the lower-change quantile need not be negative.
    ex = ns["ExplanationConfig"]()
    y = vy[:20].copy(); pm = vx[:20, 0].copy()
    a = ns["_event_labels"](pm, y, vx, vy, ex)
    b = ns["_event_labels"](pm * 1000, y * 1000, vx, vy, ex)
    assert a[3] == b[3]
    assert a[3]["delta_q25"] > 0
    assert set(a[0]).issubset({"lower_change", "middle_change", "higher_change"})


def test_episode_gaps_break_contiguity(ns):
    dates = pd.date_range("2024-01-01", periods=7, freq="6h").delete(3)
    result = ns["_episode_ids"](dates, ["same"] * 6, 6)
    np.testing.assert_array_equal(result, [1, 1, 1, 2, 2, 2])


def test_shared_calendar_grid_is_independent_of_datetime_storage_unit():
    from tsaime.calendar_statistics import calendar_frame
    dates = pd.date_range("2024-01-01", periods=12, freq="6h").delete([3, 4])
    values = np.arange(len(dates), dtype=float)
    frames = [calendar_frame(dates.as_unit(unit), values, 6) for unit in ("ns", "us", "ms")]
    for frame in frames:
        assert len(frame) == 12
        np.testing.assert_array_equal(np.flatnonzero(frame[0].isna()), [3, 4])
        np.testing.assert_allclose(frame.to_numpy(), frames[0].to_numpy(), equal_nan=True)
    bad = dates.as_unit("us").to_numpy().copy()
    bad[-1] += np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="grid"):
        calendar_frame(bad, values, 6)


def test_table_typesetting_escapes_and_wraps_without_changing_content(ns):
    formatter = ns["_tex_text_v94"]
    assert formatter("Wanshouxigong").replace(r"\-", "") == "Wanshouxigong"
    assert formatter("Reconstruction comparisons", wrap=False) == "Reconstruction comparisons"
    assert formatter("10% & PM_25") == r"10\% \& PM\_25"
    assert formatter(np.nan) == "--"
    assert formatter(0.0000023) == "2.300e-06"


def test_seasonal_summary_counts_unidentifiable_windows(ns):
    data = pd.DataFrame(dict(dataset=["A"]*4, site=["S1", "S1", "S2", "S2"], season=["DJF"]*4,
        feature=["RAIN_EVENT"]*4, horizon_h=[24]*4, AIME=[np.nan, 2., 4., 4.],
        input_varies=[False, True, True, True], output_varies=[True]*4))
    by_site, summary = ns["_seasonal_profiles"](data)
    assert summary.iloc[0].AIME == 3  # Equal site weighting, not the pooled-window median 4.
    assert summary.iloc[0].valid_windows == 3
    assert summary.iloc[0].constant_input_windows == 1
    assert summary.iloc[0].total_windows == 4


def test_manifest_rejects_changed_files_and_traversal(ns, tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("original")
    table = pd.DataFrame([dict(file=path.name, bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    assert ns["_verify_manifest"](tmp_path, table, "file") == 1
    path.write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        ns["_verify_manifest"](tmp_path, table, "file")
    with pytest.raises(ValueError, match="Unsafe"):
        ns["_safe_member"](tmp_path, "../outside")


def test_explanation_config_rejects_noninformative_blocks(ns):
    with pytest.raises(ValueError):
        ns["ExplanationConfig"](stability_block_days=(28,)).validate()
    with pytest.raises(ValueError):
        ns["ExplanationConfig"](stability_resamples=2).validate()


def test_source_reuse_rejects_wrong_config_before_reading_traces(ns, tmp_path):
    folder = tmp_path / "outputs" / "logs"
    folder.mkdir(parents=True)
    (folder / "run_info.json").write_text(json.dumps({"workflow": "APR v9.3", "config": {}}))
    with pytest.raises(ValueError, match="configuration"):
        ns["_load_v93_source"](tmp_path, ns["APRExperimentConfig"]().resolved())
