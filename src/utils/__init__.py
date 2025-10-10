"""
Utility functions and classes for the BO project.
"""

# Import from the main utils module
import logging
import random
import numpy as np
import torch

# Set up logging
logger = logging.getLogger("bo_logger") 
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:  # Avoid duplicate handlers
    logger.addHandler(handler)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

from .early_stopping import EarlyStoppingState, check_early_stopping

__all__ = ["logger", "seed_everything", "EarlyStoppingState", "check_early_stopping"]
