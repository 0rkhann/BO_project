"""
FABO BO pipeline with FABO feature selection.
"""

from src.pipelines.base_pipeline import BOPipeline
from src.feature_selection.fabo import SpearmanFABOSelector


class FABOPipeline(BOPipeline):
    """BO pipeline with FABO feature selection."""

    def selector_factory(self):
        """Create FABO feature selector with adaptive feature selection."""
        fs_config = self.cfg.get("feature_selection", {})

        # Get configuration parameters
        k = fs_config.get("fabo_k")  # None = auto-select
        threshold = fs_config.get("fabo_threshold")  # None = auto-select
        max_features = fs_config.get("fabo_max_features", 100)
        min_features = fs_config.get("fabo_min_features", 5)  # NEW
        min_correlation = fs_config.get("fabo_min_correlation", 0.1)
        cv_folds = fs_config.get("fabo_cv_folds", 5)

        return SpearmanFABOSelector(
            k=k,
            threshold=threshold,
            max_features=max_features,
            min_features=min_features,  # NEW
            min_correlation=min_correlation,
            cv_folds=cv_folds,
            logger=self.logger,
        )
