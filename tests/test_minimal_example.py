"""Contract and numerical checks for the public, dataset-free tutorial."""
from pathlib import Path

import nbformat
import numpy as np
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/tsAIME_minimal_example_v0_3_3.ipynb'


@pytest.fixture(scope='module')
def example() -> dict:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    namespace = {'__name__': '__main__'}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            exec(compile(cell.source, f'minimal-cell-{index}', 'exec'), namespace)
    return namespace


def test_example_is_clean_and_uses_public_api() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    assert 'from tsaime import' in source
    for forbidden in ('sys.path', 'read_csv', 'requests.', '/Users/', 'def fit_inverse_operator'):
        assert forbidden not in source
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None
            assert not cell.outputs


def test_forecast_training_and_inverse_evaluation_are_chronological(example: dict) -> None:
    e = example
    assert np.max(e['origins'][e['train_rows']] + max(e['horizons'])) < e['train_end']
    assert np.max(e['origins'][e['calibration']]) < np.min(e['origins'][e['evaluation']])
    np.testing.assert_allclose(e['forecasts'], e['design'] @ e['forward_weights'])
    assert e['inverse'].operator.shape == (3, 2)
    assert e['reconstructed'].shape == (300, 3)


def test_operator_and_scaling_match_calibration_only(example: dict) -> None:
    e = example
    model = e['inverse']
    x = e['X'][e['calibration']]
    y = e['forecasts'][e['calibration']]
    np.testing.assert_allclose(model.x_scaler.mean, x.mean(axis=0))
    np.testing.assert_allclose(model.y_scaler.mean, y.mean(axis=0))
    xz = (x - x.mean(axis=0)) / x.std(axis=0)
    yz = (y - y.mean(axis=0)) / y.std(axis=0)
    expected = (xz.T @ yz / len(x)) @ np.linalg.pinv(yz.T @ yz / len(x) + .01 * np.eye(2))
    np.testing.assert_allclose(model.operator, expected)
    np.testing.assert_allclose(e['terms'].iloc[:, :2].sum(axis=1), e['terms'].iloc[:, 2])
    assert np.isfinite(e['summary'].to_numpy()).all()


def test_rolling_endpoint_contract_and_operator(example: dict) -> None:
    e = example
    diagnostics = e['rolling'].diagnostics
    assert len(diagnostics) == 9
    assert (diagnostics['n_window'] == 168).all()
    last = diagnostics.iloc[-1]
    aligned = e['aligned']
    rows = aligned.loc[(aligned.Date > last.window_start_exclusive) & (aligned.Date <= last.endpoint)]
    model = e['fit_inverse_operator'](rows[e['features']], rows[e['output_names']], ridge=.01)
    actual = e['rolling'].operators.query('endpoint == @last.endpoint').pivot(index='feature', columns='output', values='coefficient')
    np.testing.assert_allclose(actual.loc[e['features'], e['output_names']], model.operator)
