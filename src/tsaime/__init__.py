"""Public API for forecast-aligned time-series AIME."""

from .operators import (
    InverseOperatorModel,
    MatrixStandardizer,
    covariance_weighted_inverse,
    fit_inverse_operator,
    marginal_cross_correlation,
    operator_cosine,
    scalar_aime_operator,
)
from .legacy import (
    RollingTSAIME,
    RollingTSAIMEConfig,
    TSAIMEResult,
    benjamini_hochberg,
    rolling_correlation,
    select_periodic_shifts,
)
from .rolling import RollingVectorTSAIME, RollingVectorTSAIMEConfig, RollingVectorTSAIMEResult
from .statistics import (
    RegressionMetrics,
    moving_block_skill_interval,
    regression_metrics,
    structured_period_shift_test,
)
from .calendar_statistics import calendar_skill_intervals, complete_week_shift_diagnostic
from .temporal import (
    ChronologicalSplit,
    QualityGate,
    audit_hourly_site,
    complete_hourly,
    evaluate_quality_gate,
    prepare_chronological_pairs,
)

__version__ = "0.3.3"

__all__ = [
    "ChronologicalSplit",
    "calendar_skill_intervals",
    "complete_week_shift_diagnostic",
    "InverseOperatorModel",
    "MatrixStandardizer",
    "QualityGate",
    "RegressionMetrics",
    "RollingVectorTSAIME",
    "RollingVectorTSAIMEConfig",
    "RollingVectorTSAIMEResult",
    "RollingTSAIME",
    "RollingTSAIMEConfig",
    "TSAIMEResult",
    "audit_hourly_site",
    "complete_hourly",
    "covariance_weighted_inverse",
    "fit_inverse_operator",
    "evaluate_quality_gate",
    "benjamini_hochberg",
    "marginal_cross_correlation",
    "moving_block_skill_interval",
    "operator_cosine",
    "prepare_chronological_pairs",
    "regression_metrics",
    "rolling_correlation",
    "scalar_aime_operator",
    "structured_period_shift_test",
    "select_periodic_shifts",
    "__version__",
]
