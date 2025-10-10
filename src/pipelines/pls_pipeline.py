"""
PLS BO pipeline with PLS feature selection.
"""

from src.pipelines.base_pipeline import BOPipeline
from src.feature_selection.pls import PLSFeatureSelector


class PLSPipeline(BOPipeline):
    """BO pipeline with PLS feature selection."""

    def selector_factory(self):
        """Create PLS feature selector."""
        n_components = self.cfg.get("feature_selection", {}).get("pls_components", 20)

        return PLSFeatureSelector(
            max_components=n_components,
            random_state=self.cfg.get("random_state", 42),
        )
