"""Compatibility imports for code written against ts-AIME 0.2."""

from .legacy import (
    RollingTSAIME,
    RollingTSAIMEConfig,
    TSAIMEResult,
    benjamini_hochberg,
    rolling_correlation,
    select_periodic_shifts,
)
from .operators import scalar_aime_operator

__all__ = [
    "RollingTSAIME",
    "RollingTSAIMEConfig",
    "TSAIMEResult",
    "benjamini_hochberg",
    "rolling_correlation",
    "scalar_aime_operator",
    "select_periodic_shifts",
]

