"""
Vanilla BO pipeline without feature selection.
"""

from src.pipelines.base_pipeline import BOPipeline


class VanillaPipeline(BOPipeline):
    """Vanilla BO pipeline - no feature selection."""

    def selector_factory(self):
        """No feature selection for vanilla pipeline."""
        return None
