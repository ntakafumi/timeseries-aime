# Time Series AIME (ts-AIME)

**Time Series AIME (ts-AIME)** interprets time-series predictions through an approximate inverse mapping from forecast outputs to input features.
It extends Approximate Inverse Model Explanations (AIME) to rolling time windows, characterizing how the feature patterns associated with predictions change over time.

This repository provides the reusable Python package **`tsaime`** and a **PM2.5 analysis notebook** that integrates ts-AIME with S-Map forecasting from Empirical Dynamic Modeling (EDM).
The notebook analyzes joint 1-, 6-, and 24-hour forecasts to reconstruct meteorological feature patterns and examine their temporal variation.

- **Package:** `tsaime` 0.3.3, imported as `tsaime`.
- **Start here:** [Minimal working notebook](notebooks/tsAIME_minimal_example_v0_3_3.ipynb), with synthetic data and no downloads.
- **PM2.5 workflow:** [v9.4 notebook](notebooks/tsAIME_APR_all_experiments_v9_4.ipynb).
- **Software license:** [PolyForm Noncommercial 1.0.0](LICENSE.txt).
- **Earlier releases:** [Zenodo archive](https://doi.org/10.5281/zenodo.20300938); see [Citation](#citation) for version-specific attribution.

## Capabilities and repository layout

- Scalar- and vector-output inverse operators with column standardization and ridge regularization.
- Rolling calendar-window estimation and time-dependent feature profiles.
- Utilities for chronological source-target alignment, data-quality checks, and dependence-preserving uncertainty estimates.
- An optional S-Map interface for out-of-sample forecasting.
- A study notebook for data acquisition, preprocessing, synthetic validation, PM2.5 experiments, and figure and table generation.

```text
src/tsaime/       Reusable mathematical and time-series methods
notebooks/       PM2.5 workflow and historical analysis notebooks
tests/           Numerical, temporal-alignment, and workflow tests
docs/            Theory, study protocols, and usage guides
DATA_SOURCES.md   Dataset identifiers, coverage, provenance, and terms
tools/           Code-release packaging utility
```

The installable package contains shared methods, not the PM2.5 study itself.
Data loaders, atmospheric variables, experiment settings, and study-specific figures remain in the notebook.
Other studies can use the package with their own aligned inputs and forecast outputs without adopting the PM2.5 workflow.

## Installation

Use a source checkout or extracted code-release archive containing `pyproject.toml` and `src/tsaime/`.
Run the commands below from that directory.
The package declares Python 3.9 or later; the PM2.5 workflow's recorded validation environment is Python 3.13 on macOS arm64.
Compatibility with other environments should be checked with the tests and a reduced notebook run.

### Core package

```bash
python -m pip install .
```

The core package depends on NumPy and pandas.
It does not require `aime-xai` or `package-v2`.
For S-Map forecasting without the notebook environment, install `'.[smap]'` instead.

### Complete notebook environment

The following commands create an isolated environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
python -m jupyter lab
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`.
Select that environment as the notebook kernel and restart the kernel after updating the package.
The `all` extra includes `pyEDM==2.5.7`, plotting, notebook, and testing dependencies.

For the recorded scientific dependency versions, use a fresh Python 3.13 environment:

```bash
python -m pip install -r requirements-v9.4.txt
python -m pip install -e . --no-deps
```

The version record is a reproduction aid, not a guarantee of identical results across operating systems or numerical libraries.
Users of the earlier `aime-xai`/pyEDM 1.x implementation should consult the [migration guide](docs/MIGRATION.md).

## Minimal example

For a runnable introduction, open the [minimal working notebook](notebooks/tsAIME_minimal_example_v0_3_3.ipynb).
It demonstrates chronological forecast training, inverse calibration, held-out reconstruction, feature profiles, a local reconstruction, and rolling calendar-window estimation.
The tutorial uses a small linear forecaster instead of S-Map so that the reusable ts-AIME API remains visible.
It requires the installed package plus the `plot` and `notebook` extras (`python -m pip install -e '.[plot,notebook]'` from the source-release directory), but no APR files or external datasets.
English explanations and short Japanese notes describe how to substitute another dataset or forecaster without changing the package.

The estimator takes two aligned matrices: input states with shape `(n_samples, n_features)` and forecast outputs with shape `(n_samples, n_outputs)`.
Each row must refer to the same forecast-origin time.
The output columns represent forecasts, not the subsequently observed targets.

This synthetic example demonstrates the API without downloading data or fitting an EDM model:

```python
import numpy as np
from tsaime import fit_inverse_operator

rng = np.random.default_rng(42)
inputs = rng.normal(size=(600, 4))
forward = np.array([
    [1.0, 0.5, 0.0, 0.0],
    [0.2, 1.0, 0.5, 0.0],
    [0.0, 0.2, 1.0, 0.5],
])
forecasts = inputs @ forward.T + 0.05 * rng.normal(size=(600, 3))

model = fit_inverse_operator(
    inputs=inputs[:400],
    outputs=forecasts[:400],
    ridge=0.01,
)

operator = model.operator                         # shape: (4, 3)
reconstructed = model.reconstruct(forecasts[400:]) # original input units
profile = np.linalg.norm(operator, axis=1)        # one score per input feature

print(operator.shape, reconstructed.shape)        # (4, 3) (200, 4)
print(profile)
```

The row norm summarizes coefficient magnitude across the output dimensions in the fitted standardized coordinates.
It is not a probability or a percentage contribution to the predicted target.
Reconstruction error and coefficient stability should be examined alongside this profile.

`RollingVectorTSAIME` and `RollingVectorTSAIMEConfig` provide the same estimator over trailing calendar windows.
They accept a timestamped DataFrame, feature-column names, and forecast-column names.
Time alignment is supplied by the analysis; the rolling estimator does not construct future targets or train a forecasting model.
See the [theory guide](docs/THEORY.md) and [study-development guide](docs/COLLABORATION.md).

## What ts-AIME explains

Within a calibration window, let $X\in\mathbb{R}^{n\times d}$ contain centered, column-standardized input states and $Y\in\mathbb{R}^{n\times q}$ contain the corresponding standardized forecast vectors.
The vector-output operator solves

$
\widehat A_\lambda
=
\arg\min_A
\left\{
n^{-1}\lVert X-YA^\top\rVert_F^2
+
\lambda\lVert A\rVert_F^2
\right\},
\qquad
\widehat A_\lambda
=
S_{XY}(S_{YY}+\lambda I_q)^\dagger.
$

where $S_{XY}=X^\top Y/n$, $S_{YY}=Y^\top Y/n$, and $\dagger$ denotes the Moore-Penrose pseudoinverse.
The implementation learns the means and scales from the calibration inputs and outputs and reuses them when reconstructing subsequent inputs.

S-Map coefficients describe a local forward relation from state coordinates to forecasts.
ts-AIME describes the reverse reconstruction relation from forecast vectors to input features over a specified time window.
These are different explanation targets: ts-AIME is neither an exact inverse of the forecasting model nor a decomposition of an individual PM2.5 prediction into additive feature contributions.
Its feature profiles describe forecast-aligned associations, not causal effects or pollution-source apportionment.

For a single nonconstant standardized output and zero regularization, the coefficients equal the input-output Pearson correlations.
With multiple outputs, the inverse operator also accounts for covariance among those outputs.
Constant variables require separate diagnostics rather than an importance interpretation.
The [theory guide](docs/THEORY.md) gives the optimization target, scalar special case, and relationship to local forward operators.

## Reproducing the PM2.5 analysis

Open [tsAIME_APR_all_experiments_v9_4.ipynb](notebooks/tsAIME_APR_all_experiments_v9_4.ipynb) and run all cells in order.
The notebook uses the local package source when available.
Package installation alone does not install the notebook or its datasets.

The workflow includes:

1. Source provenance, hourly coverage, and data-quality audits.
2. Synthetic inverse-operator recovery and temporal-tracking checks.
3. Chronological S-Map forecasting with separate training, selection, and final-evaluation periods.
4. Joint-horizon inverse reconstruction, PM2.5-history controls, coefficient stability, seasonal profiles, and event summaries.
5. PDF/PNG figures, complete CSV tables, LaTeX tables, run metadata, and an output ZIP.

The S-Map library excludes the final-evaluation period.
Rolling inverse models use preceding calibration windows and are evaluated on subsequent periods; they can therefore use earlier observations from the final-evaluation period as time advances.
The study protocol distinguishes this adaptive reconstruction from a fixed inverse model.

### Execution settings

| Setting | Default | Behavior |
|---|---|---|
| `RUN_MODE` | `publication` | Full study configuration; `smoke` is a reduced execution check |
| `EXECUTION_MODE` | `auto` | Reuse a verified compatible v9.3 foundation if available; otherwise compute it |
| `SOURCE_RUN` | `None` | Optional existing run directory containing both `outputs/` and `private_traces/` |

Set `EXECUTION_MODE = "full"` to recompute the foundation, or `"reuse"` to require a compatible saved run.
On a fresh checkout, no saved observations or run traces are included, so `auto` computes the experiment.
Downloads require internet access unless the required data are already cached locally.

For a reduced run on macOS or Linux:

```bash
TSAIME_V9_RUN_MODE=smoke TSAIME_V94_EXECUTION_MODE=full python -m jupyter lab
```

Smoke-mode outputs are for checking execution and are not the full-study results.
The [execution guide (Japanese)](docs/V9_4_RUNNING_ja.md), [protocol (Japanese)](docs/V9_4_PROTOCOL_ja.md), and [validation record (Japanese)](docs/V9_4_VALIDATION_ja.md) describe the settings and validation scope.

### Generated artifacts

Each execution creates a new directory without clearing earlier runs:

```text
results/v9_4_apr/<run-id>/
├── private_traces/         Local observation, forecast, and calibration traces
└── outputs/
    ├── figures/            PDF and PNG figures
    ├── csv/                Complete tables, TABLE_INDEX.csv, and manifests
    ├── tex/                LaTeX summaries
    ├── logs/               Configuration, environment, and provenance
    ├── README_RESULTS.md
    ├── FIGURE_GUIDE.md
    └── tsAIME_APR_v9_4_outputs_<run-id>.zip
```

Use `csv/TABLE_INDEX.csv` to locate analyses by name rather than relying on table numbers.
CSV tables retain the complete numerical results; compact LaTeX summaries record their aggregation or row counts.
Generated tables use `booktabs`, `longtable`, and `array`.
Computational checks assess execution consistency, not whether a scientific hypothesis is supported.

The output ZIP excludes downloaded raw datasets and `private_traces/`.
It can still contain figures derived from observations, so dataset terms also apply when sharing those artifacts.
Cached source files remain under `results/v9_apr/data/` or the directory set with `TSAIME_DATA_CACHE`.

## Data and provenance

The supplied study uses two regional datasets:

| Dataset | Study coverage | Identifier |
|---|---|---|
| Beijing Multi-Site Air Quality Data | 12 sites, March 2013 to February 2017 | [UCI DOI: 10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G) |
| NIES hourly atmospheric observations, Tsukuba | 1 site, January 2017 to December 2022 | [NIES DOI: 10.17595/20250418.001](https://doi.org/10.17595/20250418.001) |

[DATA_SOURCES.md](DATA_SOURCES.md) records dataset versions, access dates, source links, and observed license notices.
Run outputs record downloaded-file checksums and verification timestamps.
Raw datasets are not distributed with the code.

The recorded NIES landing-page and downloaded-file license notices differ.
Consult the provider's controlling terms before redistributing data or derived artifacts; the software license does not resolve that discrepancy.

## Tests and distribution builds

From a source checkout with the `all` dependencies installed:

```bash
python -m pytest
python -m build
python tools/build_release.py
```

The build creates the package wheel and source archive in `dist/`.
The release utility creates `release/timeseries-aime-v9.4-code-release.zip`, including source, documentation, tests, distributions, and notebook copies with execution outputs removed.
It excludes datasets, result directories, private traces, and local audits, and leaves executed notebook originals unchanged.
Release-copy hygiene tests should be run on an extracted code archive if the working notebooks contain execution outputs.

## License

This version is distributed under the [PolyForm Noncommercial License 1.0.0](LICENSE.txt).
The license permits noncommercial purposes, including the research and institutional uses specified in its terms.
Redistribution must retain the license terms and all `Required Notice:` lines.
Commercial use requires a separate license from the copyright holder.
This is source-available software with noncommercial restrictions, not an unrestricted open-source license.

Dataset licenses are separate from the software license.
Earlier archived software versions retain their accompanying notices; see the [migration guide](docs/MIGRATION.md).

## Citation

Use [CITATION.cff](CITATION.cff) for the software authors and package version, and record the exact release or commit used in an analysis.
The original AIME reference is:

Takafumi Nakanishi. *Approximate Inverse Model Explanations (AIME): Unveiling Local and Global Insights in Machine Learning Models*. IEEE Access, 11, 101020–101044, 2023. [doi:10.1109/ACCESS.2023.3314336](https://doi.org/10.1109/ACCESS.2023.3314336).

The existing [Zenodo concept DOI](https://doi.org/10.5281/zenodo.20300938) identifies the software release series.
As verified on 2026-09-06, its archived version is [`v.0.1.1`](https://doi.org/10.5281/zenodo.20300939), published on 2026-05-20.
That version-specific DOI does not archive package 0.3.3 or the v9.4 workflow.
For this code, retain the package version and exact source revision until a corresponding version-specific archive is available.

## Questions and contributions

Report reproducible issues through [GitHub Issues](https://github.com/ntakafumi/timeseries-aime/issues), including the package version, Python and dependency versions, and a minimal example.
Do not include credentials, private data, or restricted observation files.
For new applications, follow the [study-development guide](docs/COLLABORATION.md) and keep dataset-specific processing outside `src/tsaime/`.
