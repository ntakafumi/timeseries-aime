# Mathematical contract of ts-AIME 0.3.3

Version 0.3.3 preserves the 0.3.2 inverse estimator and adds a datetime-storage-unit compatibility fix to calendar-grid validation.
The v9.4 application-specific explanation controls and output generation reside in the notebook, not in the package.

## Explanation query and optimization target

ts-AIME is an inverse-operator XAI method, but it is not claimed to be the exact inverse function of the forecast model.

For centered and column-standardized matrices $X\in\mathbb{R}^{n\times d}$ and $Y\in\mathbb{R}^{n\times q}$, it solves

$$
\widehat A_\lambda
=\arg\min_A\left\{n^{-1}\lVert X-YA^\top\rVert_F^2+\lambda\lVert A\rVert_F^2\right\}.
$$

The normal equation is

$$
\widehat A_\lambda(S_{YY}+\lambda I)=S_{XY},
$$

and therefore

$$
\widehat A_\lambda=S_{XY}(S_{YY}+\lambda I)^{-1}
$$

when the regularized output covariance is nonsingular.

Thus the estimator is a transparent multivariate ridge inverse regression.

Its XAI contribution is the explanation query: identify the standardized atmospheric-state directions retained in the joint forecast vector, with explicit reconstruction error and time localization.

For the population conditional mean $m(Y)=\mathbb E[X\mid Y]$, the squared inverse risk satisfies

$$
\mathbb E\lVert X-AY\rVert^2
=\mathbb E\lVert X-m(Y)\rVert^2
+\mathbb E\lVert m(Y)-AY\rVert^2.
$$

The first term is irreducible inverse ambiguity.

The second term is the linear inverse-approximation gap.

The v9.3 workflow evaluates reconstruction on future held-out periods and compares the linear operator with a quadratic inverse-regression diagnostic using the same standardized validation loss and mean-loss penalty units.

## Objects estimated by S-Map and ts-AIME

Let $x_t\in\mathbb{R}^d$ denote a source-time state and let $y_t\in\mathbb{R}^q$ denote a vector of forecasts made from that state.

For the APR workflow, $y_t$ contains the 1-hour, 6-hour, and 24-hour PM2.5 forecasts.

S-Map locally approximates the forward relation

$$
y_t\approx F_t x_t+b_t.
$$

Its coefficient matrix $F_t\in\mathbb{R}^{q\times d}$ is a fitted local forward slope in the selected state coordinates, not necessarily the exact derivative of the full query-dependent smoother.

ts-AIME estimates a reverse reconstruction relation

$$
\widehat{x}_t=A_t y_t.
$$

The regularized empirical operator is

$$
A_t=S_{XY,t}(S_{YY,t}+\lambda I)^{-1}.
$$

Thus, $F_t$ and $A_t$ do not have the same shape or statistical target.

Transposing one does not generally recover the other.

The use of the reverse direction is not circular.

The forecast model is trained and evaluated chronologically, the inverse operator is fitted on past forecast-state pairs, and inverse fidelity is evaluated on subsequent pairs.

## Covariance-weighted bridge

When a linear forward model has state covariance $\Sigma_t$ and additive residual uncorrelated with the state, its covariance-weighted inverse is

$$
A_t^{(F)}=\Sigma_{X,t}F_t^\top(F_t\Sigma_{X,t}F_t^\top+\Sigma_{\varepsilon,t}+\lambda I)^{-1}.
$$

The APR workflow compares the empirical ts-AIME operator with this S-Map-implied inverse.

Here, $\Sigma_{\varepsilon,t}$ is the output-noise covariance in the same standardized output coordinates.

The noiseless expression is recovered only when $\Sigma_{\varepsilon,t}=0$.

The synthetic population ground truth includes the generating population state covariance and known standardized noise covariance. The sample-covariance-conditional target is reported separately.

For real rolling windows, the median local S-Map coefficient need not make residuals orthogonal to inputs. With $C=\mathrm{Cov}(X,r)$, the complete identities are $S_{XY}=\Sigma_XF^\top+C$ and $S_{YY}=F\Sigma_XF^\top+FC+C^\top F^\top+\Sigma_r$.
The v9.3 notebook reports the omitted cross-covariance magnitude, the uncorrelated-residual approximation, and the full covariance identity separately.
Restoring the omitted terms is an algebraic consistency check, not independent empirical validation.

It does not make either estimand a causal effect.

## Scalar special case

For one standardized output and $\lambda=0$,

$$
A=S_{XY}.
$$

Each component equals the Pearson correlation between one standardized input and the standardized forecast.

Therefore, scalar ts-AIME is treated as a verified special case rather than the new contribution of the APR workflow.

## Time alignment

Every future target is constructed on a complete hourly calendar before missing source-target pairs are removed.

If the PM2.5 value at $t+6$ is missing, the 6-hour target for source time $t$ remains missing.

The workflow never replaces it with the next observed row.

Train, validation, and test periods are chronological and non-overlapping.

The final target time must remain inside the same split as its source time.

## Supported interpretation

The operators describe how forecast vectors and observed state variables covary within specified time windows.

Pressure, humidity, precipitation, and wind coefficients should be described as forecast-aligned atmospheric associations.

Their predictive contribution is evaluated separately by validation-tuned, strict out-of-sample pressure and wind ablations.

They do not identify emissions sources or interventions.
