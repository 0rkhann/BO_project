"""
PCA BO pipeline with PCA feature selection.
"""

from src.pipelines.base_pipeline import BOPipeline
from src.feature_selection.pca import PCAFeatureSelector


class PCAPipeline(BOPipeline):
    """BO pipeline with PCA feature selection."""

    def selector_factory(self):
        """Create PCA feature selector with adaptive component selection."""
        fs_config = self.cfg.get("feature_selection", {})

        # Get configuration parameters
        n_components = fs_config.get("pca_components")  # None = auto-select
        max_components = fs_config.get("pca_max_components", 50)
        explained_variance_threshold = fs_config.get(
            "pca_explained_variance_threshold", 0.95
        )
        cv_folds = fs_config.get("pca_cv_folds", 5)

        return PCAFeatureSelector(
            n_components=n_components,
            max_components=max_components,
            explained_variance_threshold=explained_variance_threshold,
            cv_folds=cv_folds,
            logger=self.logger,
        )
