"""
Common utilities: logging setup, random seeds, etc.
"""

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
logger.addHandler(handler)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
