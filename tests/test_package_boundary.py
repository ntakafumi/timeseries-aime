from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "tsaime"


def test_installable_package_contains_no_apr_workflow_or_data_loader() -> None:
    assert not (PACKAGE_ROOT / "datasets.py").exists()
    assert not (PACKAGE_ROOT / "workflows" / "apr_pm25.py").exists()
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    )
    for paper_specific_term in (
        "APRExperimentConfig",
        "Beijing",
        "Tsukuba",
        "Thailand",
        "PM25_future_",
    ):
        assert paper_specific_term not in source


def test_apr_workflow_exists_only_in_the_notebook() -> None:
    assert not (ROOT / "experiments").exists()
    notebook = ROOT / "notebooks" / "tsAIME_APR_all_experiments_v9_package.ipynb"
    source = notebook.read_text(encoding="utf-8")
    assert "APRExperimentConfig" in source
    assert "PublicDataRepository" in source
    assert "run_apr_pm25" in source
