"""
XGBoost implementation for molecular property prediction.
"""

import numpy as np
from xgboost import XGBRegressor

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    def build_model(self) -> XGBRegressor:
        """Build and return an XGBoost model."""
        return XGBRegressor(
            n_estimators=self.cfg.get("n_estimators", 100),
            max_depth=self.cfg.get("max_depth", 6),
            learning_rate=self.cfg.get("learning_rate", 0.1),
            min_child_weight=self.cfg.get("min_child_weight", 1),
            subsample=self.cfg.get("subsample", 0.8),
            colsample_bytree=self.cfg.get("colsample_bytree", 0.8),
            gamma=self.cfg.get("gamma", 0),
            reg_alpha=self.cfg.get("reg_alpha", 0),
            reg_lambda=self.cfg.get("reg_lambda", 1),
            n_jobs=self.cfg.get("n_jobs", -1),
            random_state=self.cfg.get("random_state", 42),
            verbosity=0,
        )

    def get_feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """
        Return feature importance scores.

        Args:
            importance_type: Type of feature importance to compute.
                           Can be 'weight', 'gain', or 'cover'.
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.get_booster().get_score(importance_type=importance_type)
