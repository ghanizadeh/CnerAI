"""
foam_performance_calc_utility
------------------------------
Public surface of the foam performance calculation package.

Import the pipeline entry point and config schema directly from the package:

    from foam_performance_calc_utility import run_pipeline, PipelineConfig

Everything else is available via its submodule if needed.
"""

from .pipeline import run_pipeline
from .schemas import PipelineConfig, PipelineResult, ColumnMapping

__all__ = [
    "run_pipeline",
    "PipelineConfig",
    "PipelineResult",
    "ColumnMapping",
]
