# Migrating from earlier ts-AIME releases

## Package and workflow versions

The Python package version and the PM2.5 notebook version identify different components.
Package `tsaime` 0.3.3 supplies the shared methods; workflow v9.4 specifies the PM2.5 experiment.

| Earlier implementation | Package 0.3.3 and workflow v9.4 |
|---|---|
| Root-level `tsaime/` and `setup.py` | `src/tsaime/` and `pyproject.toml` |
| `aime-xai` plus pyEDM 1.x | NumPy and pandas for the core; optional `pyEDM==2.5.7` |
| Notebook-oriented scalar rolling analysis | Shared scalar/vector APIs and a separate, notebook-owned PM2.5 study |
| Root-level analysis notebooks | `notebooks/tsAIME_APR_all_experiments_v9_4.ipynb` for the current study |

Do not assume that an earlier notebook is a drop-in client of the current package.
The legacy scalar API retained in 0.3.3 is documented by `src/tsaime/legacy.py` and `tests/test_legacy_api.py`; it does not establish compatibility with every historical notebook.
Earlier analysis results remain records of their original code and protocol.

## Installation and imports

Use a fresh environment and install the current source tree with:

```bash
python -m pip install -e '.[all]'
```

After restarting the notebook kernel, confirm the imported implementation:

```python
import tsaime
print(tsaime.__version__)
print(tsaime.__file__)
```

The version should be `0.3.3`, and the path should identify the intended source tree or installed package.
An obsolete root-level `tsaime/` directory in the working directory can shadow the installed package.
Installing a new wheel does not remove such a directory from an existing checkout.

When updating an older checkout, preserve historical files in version control or a separate legacy directory, and use the current source layout at the repository root.
Do not combine an old `setup.py` and root-level `tsaime/` with the new `pyproject.toml` and `src/tsaime/` as competing active installations.
The code-release archive contains a `timeseries-aime/` directory whose contents form the current repository layout; `codev9` is not a required directory name.
No migration step requires deleting prior experimental results.

## Explanation outputs

`fit_inverse_operator()` accepts aligned inputs and forecast vectors and returns a standardized output-to-input operator.
`reconstruct()` applies the fitted standardization and returns reconstructed inputs in their original units.
`RollingVectorTSAIME` estimates the operator on trailing calendar windows.

For one nonconstant output with zero regularization, standardized inverse coefficients equal Pearson correlations.
For multiple outputs, the estimator accounts for covariance among the outputs.
The PM2.5 study evaluates the joint 1-, 6-, and 24-hour forecast vector.

Names such as `local_contributions` in the legacy scalar API should not be substituted for the v9.4 reconstruction terms.
In v9.4, each horizon-specific term contributes to reconstruction of an input feature, not to an additive decomposition of the PM2.5 prediction.
See [THEORY.md](THEORY.md) for the mathematical definitions.

## Historical archives and notices

The [Zenodo concept DOI](https://doi.org/10.5281/zenodo.20300938) identifies the release series.
On 2026-09-06, it resolved to the archive for `v.0.1.1`, released on 2026-05-20, with version DOI [10.5281/zenodo.20300939](https://doi.org/10.5281/zenodo.20300939).
This existing archive is not a deposit of package 0.3.3 or workflow v9.4.

The historical GitHub `LICENSE.txt` contains BSD-style terms followed by an academic/noncommercial restriction, while the existing Zenodo record labels that archive CC BY 4.0.
These are recorded historical notices, not a claim that they are equivalent or that their inconsistency has been resolved.
Retain the accompanying notices when preserving earlier releases.
The current source tree uses the [PolyForm Noncommercial License 1.0.0](../LICENSE.txt); this documentation update does not alter prior archives.

Public records checked on 2026-09-06; GitHub links are pinned to the inspected commit so that later repository updates do not change these historical references:

- [GitHub README](https://github.com/ntakafumi/timeseries-aime/blob/7857d3d12fba218680828cb420cd85b38982999a/README.md).
- [GitHub setup.py](https://github.com/ntakafumi/timeseries-aime/blob/7857d3d12fba218680828cb420cd85b38982999a/setup.py).
- [GitHub license](https://github.com/ntakafumi/timeseries-aime/blob/7857d3d12fba218680828cb420cd85b38982999a/LICENSE.txt).
- [Zenodo version record](https://zenodo.org/records/20300939).
