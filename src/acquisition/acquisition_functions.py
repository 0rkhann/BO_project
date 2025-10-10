"""
Extended acquisition functions for Bayesian Optimization.

This module provides a unified interface for various acquisition functions
and their batch variants.
"""

import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model


def create_acquisition_function(
    model: Model,
    best_f: float,
    acq_name: str,
    beta: float = 1.0,
    batch_size: int = 1,
    maximize: bool = False,
) -> AcquisitionFunction:
    """
    Create acquisition function with support for both single and batch variants.

    Args:
        model: BoTorch model
        best_f: Best observed value
        acq_name: One of ['ei', 'pi', 'ucb']
        beta: Exploration weight for UCB
        batch_size: Number of points to acquire (>1 enables batch variants)
        maximize: Whether to maximize (True) or minimize (False)
    """
    best_f = torch.tensor([best_f])

    if batch_size > 1:
        if acq_name.lower() == "ei":
            return qExpectedImprovement(
                model=model,
                best_f=best_f,
                maximize=maximize,
            )
        elif acq_name.lower() == "ucb":
            return qUpperConfidenceBound(
                model=model,
                beta=beta,
                maximize=maximize,
            )
        else:
            raise ValueError(
                f"Batch variant not available for {acq_name}. "
                "Use 'ei' or 'ucb' for batch optimization."
            )
    else:
        if acq_name.lower() == "ei":
            return LogExpectedImprovement(
                model=model,
                best_f=best_f,
                maximize=maximize,
            )
        elif acq_name.lower() == "pi":
            return ProbabilityOfImprovement(
                model=model,
                best_f=best_f,
                maximize=maximize,
            )
        elif acq_name.lower() == "ucb":
            return UpperConfidenceBound(
                model=model,
                beta=beta,
                maximize=maximize,
            )
        else:
            raise ValueError(
                f"Unknown acquisition function: {acq_name}. "
                "Use one of: ['ei', 'pi', 'ucb']"
            )
