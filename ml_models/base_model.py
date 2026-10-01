"""
Base class for ML models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    def __init__(self, config: dict):
        self.cfg = config
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    @abstractmethod
    def build_model(self):
        """Build and return the model instance."""
        pass

    def preprocess_data(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input data."""
        if y is not None:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            return X_scaled, y_scaled
        else:
            X_scaled = self.scaler_X.transform(X)
            return X_scaled

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5
    ) -> Dict[str, float]:
        """Perform k-fold cross validation."""
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.cfg.get("random_state", 42),
        )

        metrics = {"rmse_train": [], "rmse_val": [], "r2_train": [], "r2_val": []}

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Preprocess data
            X_train_scaled, y_train_scaled = self.preprocess_data(X_train, y_train)
            X_val_scaled = self.scaler_X.transform(X_val)

            # Train model
            self.model = self.build_model()
            if hasattr(self.model, 'fit'):
                # Sklearn-style model
                self.model.fit(X_train_scaled, y_train_scaled)
            else:
                # Custom model (like neural network) - use the train method
                self.train(X_train, y_train)

            # Make predictions
            if hasattr(self.model, 'predict'):
                # Sklearn-style prediction
                y_train_pred = self.scaler_y.inverse_transform(
                    self.model.predict(X_train_scaled).reshape(-1, 1)
                ).ravel()
                y_val_pred = self.scaler_y.inverse_transform(
                    self.model.predict(X_val_scaled).reshape(-1, 1)
                ).ravel()
            else:
                # Custom prediction method
                y_train_pred = self.predict(X_train)
                y_val_pred = self.predict(X_val)

            # Calculate metrics
            metrics["rmse_train"].append(
                np.sqrt(mean_squared_error(y_train, y_train_pred))
            )
            metrics["rmse_val"].append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
            metrics["r2_train"].append(r2_score(y_train, y_train_pred))
            metrics["r2_val"].append(r2_score(y_val, y_val_pred))

        # Calculate mean and std for each metric
        results = {}
        for metric in metrics:
            results[f"{metric}_mean"] = np.mean(metrics[metric])
            results[f"{metric}_std"] = np.std(metrics[metric])

        return results

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the entire dataset."""
        X_scaled, y_scaled = self.preprocess_data(X, y)
        self.model = self.build_model()
        self.model.fit(X_scaled, y_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for new data."""
        X_scaled = self.preprocess_data(X)
        y_pred_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    def save_model(self, path: Path):
        """Save the model and scalers."""
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler_X": self.scaler_X,
                "scaler_y": self.scaler_y,
                "config": self.cfg,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: Path):
        """Load a saved model."""
        import joblib

        data = joblib.load(path)
        instance = cls(data["config"])
        instance.model = data["model"]
        instance.scaler_X = data["scaler_X"]
        instance.scaler_y = data["scaler_y"]
        return instance
