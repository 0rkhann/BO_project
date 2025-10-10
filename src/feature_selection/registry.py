# registry.py

from .vanilla        import VanillaSelector
from .pls            import PLSFeatureSelector
from .fabo           import SpearmanFABOSelector
from .pca            import PCAFeatureSelector
from .opls           import OPLSFeatureSelector

_SELECTOR_MAP = {
    "vanilla": VanillaSelector,
    "pls":     PLSFeatureSelector,
    "fabo":    SpearmanFABOSelector,
    "pca":     PCAFeatureSelector,
    "opls":    OPLSFeatureSelector,
}

def get_selector(name: str, **kwargs):
    """
    Return an instance of the feature‐selector corresponding to `name`.
    Valid names: vanilla, pls, fabo, pca, opls.
    """
    try:
        cls = _SELECTOR_MAP[name]
    except KeyError:
        raise ValueError(f"Unknown feature‐selector '{name}'")
    return cls(**kwargs)
