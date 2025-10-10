"""
Configuration defaults and environment overrides.
"""

import os

# Bayesian optimization parameters
defaults = {
    "n_initial": int(os.getenv("BO_N_INITIAL", 10)),
    "n_iter": int(os.getenv("BO_N_ITER", 100)),
    "pls_components": None,               # if None, PLS auto-selects via CV
    "pls_max_components": int(os.getenv("BO_PLS_MAX", 10)),
    "pls_cv": int(os.getenv("BO_PLS_CV", 5)),
    "fabo_threshold": float(os.getenv("BO_FABO_THRESHOLD", 0.1)),
    "fabo_k": None,                       # if None, FABO auto-selects via elbow
    "random_state": int(os.getenv("BO_RANDOM_STATE", 42)),
}
