# Time-Series AIME (ts-AIME)
[![DOI](https://zenodo.org/badge/1244102225.svg)](https://doi.org/10.5281/zenodo.20300938)

**Time-Series AIME (ts-AIME)** combines the **S-Map** from **Empirical Dynamic Modeling (EDM)** with 
**Approximate Inverse Model Explanations (AIME)** [oai_citation:0‡Approximate_Inverse_Model_Explanations_AIME_Unveiling_Local_and_Global_Insights_in_Machine_Learning_Models (8).pdf](file-service://file-3pLPafHrDkvQZCdyEPCKNg) to 
**visualize and test feature contributions (global/local) in time series data through rolling analysis.

AIME is a novel XAI method that constructs an approximate inverse mapping “from the output side (predictions/estimates) to the input side (features),” 
providing both **global** and **local** feature importance within the same framework.
ts-AIME applies this concept **along the time axis**, enabling it to capture contributions involving seasonality and state dependency.

---

## Features

- Integrates predictions via **EDM (pyEDM)**'s S-Map with **AIME**'s inverse operator \( A^\dagger \)
- Visualizes temporal changes in contributions using a **rolling window**
- Calculates **global contributions** (importance across the entire window) and **local contributions** (importance at each time point)
- Calculates **95% envelope** and **p-value** from **null distribution via cyclic shift**
- Extracts significant intervals using **Benjamini–Hochberg FDR**

---

## Installation

### Dependencies
- Python 3.9+
- `pyEDM==1.14.0.2`
- `aime-xai` (core AIME implementation; uses `explainer.A_dagger`)
- `numpy`, `pandas`, `matplotlib`

```bash
pip install -U pyEDM==1.14.0.2 aime-xai numpy pandas matplotlib

# How to Interpret the Output
•    Periods where the solid line (global contribution) extends significantly beyond the envelope indicate "significant contribution".
•    Local contributions represent end-of-window or daily contributions. Spikes appear on days with extreme events.
•    By extracting significant intervals (FDR<0.05), we can quantify seasonal contribution shifts.

⸻

# Important Notes
- AIME assumes ((X, \hat{Y})) is normalized before constructing (A^\dagger).
- Caution is needed regarding stability when windows are too small or outputs are close to constant, as (\hat{Y} \hat{Y^\top}) can become singular.
- With few permutations (R), p-value resolution is coarse. Typically 500–1000 permutations are recommended.
- Using weekly shift preservation (preserve_weekly=True) constructs a stricter null hypothesis without disrupting day-of-week patterns.
⸻

# Citation

For academic use, please cite as follows:
•	AIME
Takafumi Nakanishi,
Approximate Inverse Model Explanations (AIME): Unveiling Local and Global Insights in Machine Learning Models,
IEEE Access, vol. 11, pp. 101020-101044, 2023. DOI: 10.1109/ACCESS.2023.3314336 ￼
•	ts-AIME
Takafumi Nakanishi et al., ts-AIME: Time-Series AIME on top of EDM (pyEDM), 2025, software package.
