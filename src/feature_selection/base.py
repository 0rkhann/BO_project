from abc import ABC, abstractmethod
import torch
import numpy as np


class FeatureSelector(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def transform(self, X): ...

    def prepare(
        self,
        train_X: "np.ndarray",
        train_y: "np.ndarray",
        pool_X: "np.ndarray",
        test_X: "np.ndarray",
    ):
        """
        Default: no feature selection.
        Just package numpy arrays into torch tensors.
        """
        return (
            torch.tensor(train_X, dtype=torch.double),
            torch.tensor(pool_X, dtype=torch.double),
            torch.tensor(test_X, dtype=torch.double),
        )

    def update(
        self,
        train_X: "np.ndarray",
        train_y: "np.ndarray",
        pool_X: "np.ndarray",
        test_X: "np.ndarray",
    ):
        """
        Default update for BO loop: delegate to prepare().
        Returns (train_x, pool_x, test_x) as torch.DoubleTensors.
        """
        return self.prepare(train_X, train_y, pool_X, test_X)
