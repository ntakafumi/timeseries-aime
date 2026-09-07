from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "tsAIME_APR_all_experiments_v9_package.ipynb"


def test_notebook_contains_apr_code_but_not_tsaime_implementations() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "from experiments" not in source
    assert "class APRExperimentConfig" in source
    assert "class PublicDataRepository" in source
    assert "def prepare_pm25_multihorizon_pairs" in source
    assert "def apply_pm25_policy" in source
    assert "def _meteorology_ablation" in source
    assert "def _synthetic_time_local_tracking" in source
    assert "def _ridge_inverse_objective_diagnostic" in source
    assert "def _fit_quadratic_inverse" in source
    assert "def _inverse_output_subset_diagnostic" in source
    assert "def _rolling_inverse_oos" in source
    assert "def run_apr_pm25" in source
    assert "RESULT = run_apr_pm25" in source
    assert "def fit_inverse_operator" not in source
    assert "def run_smap_oos" not in source
    assert "def moving_block_skill_interval" not in source
    assert "/Users/" not in source
    assert "10.24432/C5RK5G" in source
    assert "10.17595/20250418.001" in source
    assert "latex_max_rows=" not in source
    assert "Time-Series AIME (ts-AIME)" in source
    assert "v9.2_multisite_oos_inverse_operator" in source
    assert '"package_version": "0.3.1"' in source


def test_every_notebook_code_cell_compiles() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")


def test_notebook_is_clean_for_github() -> None:
    # v9.2/v9.3 are preserved executed research records, not release notebooks.
    notebook = nbformat.read(ROOT / "notebooks" / "tsAIME_APR_all_experiments_v9_4.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
    assert notebook.metadata["tsaime_v9"]["apr_code_location"] == "notebook"
    assert notebook.metadata["tsaime_v9"]["package_contains_paper_workflow"] is False
