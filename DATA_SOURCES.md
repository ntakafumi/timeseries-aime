# Data sources and redistribution record

This file records the sources used by the APR PM2.5 workflow.

It does not replace the data providers' controlling terms.

The terms must be checked again immediately before public release because licenses and landing pages can change.

## Required source 1: Beijing

- Dataset: Beijing Multi-Site Air Quality Data
- Provider: UCI Machine Learning Repository / Beijing Municipal Environmental Monitoring Center
- DOI: `10.24432/C5RK5G`
- Version: UCI dataset 501 download snapshot
- Period used: 2013-03-01 through 2017-02-28
- Sites: 12
- Resolution: hourly
- Landing page: <https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata>
- Access date: 2026-09-04
- License displayed on 2026-09-04: CC BY 4.0
- Redistribution policy in this project: raw data are not committed or included in result ZIP files

## Required source 2: Tsukuba

- Dataset: Atmospheric Environmental Regional Observation System hourly observations
- Provider: National Institute for Environmental Studies (NIES), Japan
- DOI: `10.17595/20250418.001`
- Version: 1.0
- Period used: 2017-01-01 through 2022-12-31
- Site: Tsukuba
- Resolution: hourly
- Landing page: <https://db.cger.nies.go.jp/MD/10.17595/20250418.001.html.en>
- Access date: 2026-09-04
- License displayed on the landing page on 2026-09-04: CC BY-NC-ND 4.0
- Notice observed in downloaded text headers on 2026-09-04: CC BY 4.0
- Required release action: do not infer that the discrepancy is resolved; verify the publisher's current controlling terms and cite the dataset as requested by NIES
- Redistribution policy in this project: no raw redistribution

Negative PM2.5 observations are retained in the prespecified main analysis as provider-reported measurements.

The notebook separately repeats the strict out-of-sample analysis after treating negative values as missing and after replacing them with zero.

## Optional Thai extension

The Thai extension combines OpenAQ observations with NASA POWER meteorology.

It is disabled by default and is not required for the Beijing-plus-Tsukuba study.

The v9.3 configuration rejects enabling this extension without a separately reviewed protocol. A future extension must verify and record its exact observations, station identity, coverage, DOI availability, version/snapshot, and current source terms before results are used in a manuscript.

No optional raw data are redistributed.

## Execution-level evidence

Each v9.3 notebook run writes the following under `outputs/`; use `csv/TABLE_INDEX.csv` to identify the analysis names:

- `csv/table26_data_sources.csv`: DOI, version, recorded access date, license status, analytical role, and redistribution decision;
- `csv/table27_license_audit.csv`: the recorded landing-page license label and the notice parsed from each downloaded NIES file header;
- `csv/table28_download_provenance.csv`: source URL, cache path, bytes, SHA-256, cache status, cache-file timestamp, and run verification timestamp;
- `DATA_AVAILABILITY.txt`: a draft Data Availability paragraph, subject to resolution of the NIES notice discrepancy;
- `logs/run_info.json`: package and dependency versions, available Git state, package-source SHA-256, and notebook-code SHA-256.

The 2026-09-04 source-access record is not a claim that the landing-page terms are verified afresh on every cached run. File-level verification timestamps identify what the current execution actually checked.

The public repository and result ZIP exclude the cached raw datasets.
