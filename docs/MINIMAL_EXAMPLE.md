# Minimal notebook: usage and validation

Start with [tsAIME_minimal_example_v0_3_3.ipynb](../notebooks/tsAIME_minimal_example_v0_3_3.ipynb).
It has eight code cells (80 code lines), with English explanations and short Japanese notes.

## Dependencies

Install the source release with `python -m pip install -e '.[plot,notebook]'` from the directory containing `pyproject.toml`.
Launch Jupyter in that environment, open the notebook, and run all cells.
The notebook checks that the imported `tsaime` version is 0.3.3.
It also works with the 0.3.3 wheel plus the plotting and notebook dependencies.
No GitHub installation, dataset download, APR result, `package-v2`, or source-path injection is used.

## What the example demonstrates

1. Generate three synthetic input series and a measured target.
2. Fit a two-horizon linear forecaster on early observations, keeping every training target before the split boundary.
3. Fit the public `fit_inverse_operator` API on later, aligned state–forecast pairs.
4. Inspect signed coefficients, row-norm feature profiles, and reconstruction RMSE on a subsequent period.
5. Decompose one reconstructed input state into horizon-specific terms.
6. Use `RollingVectorTSAIME` to obtain descriptive trailing-window profiles.

The inverse calibration and evaluation each contain 300 origins; the forecaster uses 594 origins after target-boundary exclusion.
The rolling analysis produces nine operators, each using 168 hourly pairs.
This tutorial uses ordinary least squares, not S-Map. Forecasting can be replaced without changing the package's inverse implementation.

The two horizons sum into reconstructed input coordinates, not into target predictions.
Coefficient norms are not probabilities or causal effects.
Good input reconstruction and good future-target forecasting are separate properties; this tutorial's numerical check evaluates the former.
The synthetic forecaster can retain input information even when its future-target prediction is imperfect.

## Recorded validation

Validated on 2026-09-07 with Python 3.13, tsaime 0.3.3, NumPy 2.4.4, pandas 3.0.2, Matplotlib 3.10.9, and nbformat 5.10.4.
All eight code cells completed in a fresh Jupyter kernel using an isolated installation of the release wheel.
No local source directory was added to the notebook's import path.

With seed 42, the displayed values were:

| Feature | Inverse row norm | Reconstruction RMSE | Mean-only RMSE | RMSE reduction (%) |
|---|---:|---:|---:|---:|
| slow_signal | 0.857 | 0.080 | 1.111 | 92.776 |
| fast_signal | 1.848 | 0.147 | 1.029 | 85.750 |
| unrelated_feature | 0.112 | 0.942 | 0.952 | 1.130 |

These are tutorial outputs, not atmospheric observations or evidence for the APR manuscript.
Small finite-sample associations with the unrelated feature are expected; the example does not force its fitted coefficient to zero.
The row norm of `fast_signal` is larger, but this does not make it a universally more important predictor than `slow_signal`.

Four tests verify the notebook's clean public form, chronological boundaries, calibration-only scaling and operator formula, and rolling-window membership.
The clean release staging copy passed all 59 repository tests, including these four.
The figure was also generated in Jupyter and visually inspected.
Other operating systems and dependency versions were not tested in this validation.

## Starting another paper

Replace the data and forecaster in the notebook; keep study-specific choices outside `src/tsaime/`.
Specify forecast origins, target times, chronological selection and evaluation periods, input units, and the reconstruction question before evaluating results.
Add application-appropriate controls, uncertainty estimates, sensitivity checks, data provenance and independent evaluation.
The tutorial is an executable starting point, not a complete publication protocol.
