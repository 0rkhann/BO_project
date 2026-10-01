"""
Random Forest implementation for molecular property prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    def build_model(self) -> RandomForestRegressor:
        """Build and return a Random Forest model."""
        return RandomForestRegressor(
            n_estimators=self.cfg.get("n_estimators", 100),
            max_depth=self.cfg.get("max_depth", None),
            min_samples_split=self.cfg.get("min_samples_split", 2),
            min_samples_leaf=self.cfg.get("min_samples_leaf", 1),
            max_features=self.cfg.get("max_features", "sqrt"),
            n_jobs=self.cfg.get("n_jobs", -1),
            random_state=self.cfg.get("random_state", 42),
        )

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.feature_importances_
