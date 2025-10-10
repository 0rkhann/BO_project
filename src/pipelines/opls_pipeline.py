"""
OPLS BO pipeline with OPLS feature selection.
"""

from src.pipelines.base_pipeline import BOPipeline
from src.feature_selection.opls import OPLSFeatureSelector


class OPLSPipeline(BOPipeline):
    """BO pipeline with OPLS feature selection."""

    def selector_factory(self):
        """Create OPLS feature selector."""
        # Get OPLS components from config, with sensible defaults
        opls_config = self.cfg.get("feature_selection", {})
        n_components = opls_config.get("opls_components")

        # If n_components is None, use adaptive selection with reasonable limits
        if n_components is None:
            # For adaptive selection, use reasonable defaults
            max_predictive = 10  # Reasonable default for predictive components
            max_orthogonal = 3  # Reasonable default for orthogonal components
        else:
            # Use the specified number of components
            max_predictive = n_components
            max_orthogonal = min(
                3, n_components // 2
            )  # Orthogonal components should be fewer

        return OPLSFeatureSelector(
            max_predictive=max_predictive,
            max_orthogonal=max_orthogonal,
            cv=opls_config.get("opls_cv_folds", 5),
            random_state=self.cfg.get("random_state", 42),
            scale=True,
            logger=self.logger,
        )
