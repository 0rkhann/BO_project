"""
Acquisition functions for Bayesian Optimization.
"""

import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qUpperConfidenceBound,
    qLogExpectedImprovement,
)
from botorch.models.model import Model


def make_acquisition(
    model: Model,
    best_f: float,
    acq_name: str = "ei",
    beta: float = 1.0,
    batch_size: int = 1,
    maximize: bool = False,  # False for minimization (binding energy)
):
    """
    Build an acquisition function.

    Args:
        model: Fitted BoTorch model
        best_f: Current best observed value
        acq_name: One of ['ei', 'pi', 'ucb']
        beta: Exploration parameter for UCB
        batch_size: Number of points to acquire (>1 for batch variants)
        maximize: Whether to maximize (True) or minimize (False) -
                 False for binding energy optimization

    Returns:
        Acquisition function
    """
    name = acq_name.lower()
    best_f = torch.tensor([best_f])

    if batch_size > 1:
        # Batch variants
        if name == "ei":
            return qLogExpectedImprovement(
                model=model, best_f=best_f, maximize=maximize
            )
        elif name == "ucb":
            return qUpperConfidenceBound(model=model, beta=beta, maximize=maximize)
        else:
            raise ValueError(
                f"Batch variant not available for '{name}'. "
                "Use 'ei' or 'ucb' for batch optimization."
            )
    else:
        # Single-point variants
        if name == "ei":
            # Use LogExpectedImprovement for better numerical stability
            return LogExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        elif name == "pi":
            return ProbabilityOfImprovement(
                model=model, best_f=best_f, maximize=maximize
            )
        elif name == "ucb":
            return UpperConfidenceBound(model=model, beta=beta, maximize=maximize)
        else:
            raise ValueError(f"Unknown acquisition function '{acq_name}'")
