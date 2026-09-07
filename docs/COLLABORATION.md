# Collaboration workflow

## Starting a new study

1. Create a new analysis directory outside `src/tsaime`.
2. Retain the generic package without copying its functions into the notebook.
3. Define the state variables, output vector, forecast model, chronological splits, and quality gates before examining test results.
4. Build future targets on the complete calendar.
5. Select forecasting and inverse-operator hyperparameters using validation data only.
6. Keep the test period outside the S-Map library and all other model-selection steps.
7. Save the complete configuration, package version, input checksums, tables, and figures.

## Division of responsibilities

The shared package owns:

- mathematical operators;
- time alignment and data audits;
- confidence intervals and structured nulls;
- output manifests and ZIP creation.

Each paper-specific notebook owns:

- scientific questions;
- data-source eligibility;
- state and target definitions;
- chronological boundaries;
- manuscript figures and interpretations.

This separation prevents a new paper from silently changing the mathematical definition used by an earlier paper.

## Adding another location

A harmonized site table should contain:

```text
Date, PM25, TEMP, RH, PRESS, RAIN, WIND_U, WIND_V
```

`Date` should represent the local observation hour without duplicated timestamps.

`WIND_U` and `WIND_V` must document whether the source direction is meteorological FROM-direction.

Raw data licenses and station metadata remain the responsibility of the paper-specific workflow.

## Pull-request checks

Run:

```bash
python -m pytest
python -m build
```

Execute the notebook in `smoke` mode separately before merging changes to paper-specific cells.

Do not commit downloaded raw datasets or credentials.

Review `table17_computational_readiness.csv` and the generated figures before merging changes that affect a study workflow. A computational PASS is not a journal-acceptance guarantee.

## License and sharing

This repository is licensed under the PolyForm Noncommercial License 1.0.0.

Collaborators may use, study, modify, and share the software only for purposes permitted by that license.

Every copy must retain `LICENSE.txt` and all plain-text lines beginning with `Required Notice:`.

Commercial use requires a separate license from the copyright holder.
