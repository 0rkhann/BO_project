"""
Early stopping utilities for optimization.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class EarlyStoppingState:
    """State for early stopping."""

    best_value: float
    n_no_improvement: int = 0
    window_size: int = 10
    history: List[float] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def update(self, new_value: float) -> None:
        """Update state with new value."""
        self.history.append(new_value)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if new_value < self.best_value:
            self.best_value = new_value
            self.n_no_improvement = 0
        else:
            self.n_no_improvement += 1


def check_early_stopping(
    state: EarlyStoppingState,
    patience: int = 10,
    min_delta: float = 1e-4,
    min_iterations: int = 20,
) -> Optional[str]:
    """
    Check if optimization should stop.

    Returns:
        None if should continue, stop reason string otherwise
    """
    # Too few iterations
    if len(state.history) < min_iterations:
        return None

    # No improvement for too long
    if state.n_no_improvement >= patience:
        return "No improvement for too long"

    # Check if recent values are stable
    if len(state.history) >= state.window_size:
        recent_std = np.std(state.history)
        if recent_std < min_delta:
            return "Values have stabilized"

    return None
