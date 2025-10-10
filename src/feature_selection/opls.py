import numpy as np
import torch
from typing import Union
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from .base import FeatureSelector

# Optional dependency - graceful handling
try:
    from pyopls.opls import OPLS

    HAS_PYOPLS = True
except ImportError:
    HAS_PYOPLS = False
    OPLS = None


class OPLSFeatureSelector(FeatureSelector):
    def __init__(
        self,
        cv: int = 5,
        max_predictive: int = 15,
        max_orthogonal: int = 5,
        min_predictive: int = 2,
        random_state=None,
        scale: bool = True,
        logger=None,
        **kwargs,
    ):
        """
        cv              : number of folds for cross-validation
        max_predictive  : maximum number of predictive (PLS) components to try
        max_orthogonal  : maximum number of orthogonal (OPLS filter) components to try
        min_predictive  : minimum number of predictive components (prevents too few)
        random_state    : for reproducibility
        scale           : whether to standardize X before fitting
        logger          : optional logger
        **kwargs        : catch legacy args
        """
        super().__init__()
        self.cv = cv
        # Ensure max_predictive, max_orthogonal, and min_predictive are integers
        self.max_predictive = max_predictive if max_predictive is not None else 15
        self.max_orthogonal = max_orthogonal if max_orthogonal is not None else 5
        self.min_predictive = min_predictive if min_predictive is not None else 2
        self.random_state = random_state
        self.scale = scale
        self.logger = logger

        # to be set in fit()
        self.scaler = None
        self.ortho_filter = None
        self.pls = None
        self.n_predictive = None
        self.n_orthogonal = None

    def fit(self, X: np.ndarray, y) -> "OPLSFeatureSelector":
        """
        Fit OPLS feature selector. Falls back to PLS if pyopls is not available.
        Grid-search CV over (predictive, orthogonal) component pairs and fit final models.
        """
        if not HAS_PYOPLS:
            if self.logger:
                self.logger.warning(
                    "pyopls not available, falling back to PLS-only mode"
                )
            # Fall back to PLS regression only
            return self._fit_pls_fallback(X, y)
        y_arr = np.asarray(y, dtype=float)

        # optional scaling
        if self.scale:
            self.scaler = StandardScaler()
            X_proc = self.scaler.fit_transform(X)
        else:
            X_proc = X.copy()

        n_samples, n_feats = X_proc.shape
        max_pred = min(self.max_predictive, n_feats - 1)
        max_orth = min(self.max_orthogonal, n_feats - 1)

        best_score = -np.inf
        best_pred, best_orth = 1, 0
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # nested CV search
        for n_pred in range(1, max_pred + 1):
            for n_orth in range(0, max_orth + 1):
                scores = []
                for train_idx, val_idx in kf.split(X_proc):
                    Xtr = X_proc[train_idx]
                    Xv = X_proc[val_idx]
                    ytr = y_arr[train_idx]
                    yv = y_arr[val_idx]

                    # orthogonal filtering if requested
                    if n_orth > 0:
                        n_orth_safe = min(n_orth, len(Xtr) - 1, Xtr.shape[1] - 1)
                        if n_orth_safe < 1:
                            scores.append(-np.inf)
                            continue
                        opls = OPLS(n_components=n_orth_safe, scale=self.scale)
                        Xtr_filt = opls.fit_transform(Xtr, ytr)
                        Xv_filt = opls.transform(Xv)
                    else:
                        Xtr_filt, Xv_filt = Xtr, Xv

                    # predictive PLS - ensure n_pred doesn't exceed training samples
                    n_pred_safe = min(n_pred, len(Xtr) - 1, Xtr_filt.shape[1] - 1)
                    if n_pred_safe < 1:
                        scores.append(-np.inf)
                        continue

                    try:
                        pls = PLSRegression(n_components=n_pred_safe)
                        pls.fit(Xtr_filt, ytr)
                        yv_pred = pls.predict(Xv_filt).ravel()

                        # compute R^2
                        ss_res = np.sum((yv - yv_pred) ** 2)
                        ss_tot = np.sum((yv - yv.mean()) ** 2)
                        if ss_tot > 0:
                            scores.append(1 - ss_res / ss_tot)
                        else:
                            scores.append(0.0)
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"PLS failed in OPLS CV fold: {e}")
                        scores.append(-np.inf)

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_pred, best_orth = n_pred, n_orth

        # final fit on full data
        self.n_predictive = best_pred
        self.n_orthogonal = best_orth
        
        # Enforce minimum predictive components
        self.n_predictive = max(self.min_predictive, self.n_predictive)

        # fit orthogonal filter if needed
        if best_orth > 0:
            self.ortho_filter = OPLS(n_components=best_orth, scale=self.scale)
            X_filt_all = self.ortho_filter.fit_transform(X, y_arr)
        else:
            self.ortho_filter = None
            X_filt_all = (
                self.scaler.transform(X)
                if self.scale and self.scaler is not None
                else X
            )

        # fit predictive PLS on filtered data
        self.pls = PLSRegression(n_components=best_pred)
        self.pls.fit(X_filt_all, y_arr)

        if self.logger:
            self.logger.info(
                f"OPLSFeatureSelector fit → predictive={best_pred}, orthogonal={best_orth}, CV_score={best_score:.4f}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data to predictive score space."""
        if self.scale and self.scaler is not None:
            X_proc = self.scaler.transform(X)
        else:
            X_proc = X

        if (
            self.n_orthogonal is not None
            and self.n_orthogonal > 0
            and self.ortho_filter is not None
        ):
            X_filt = self.ortho_filter.transform(X_proc)
        else:
            X_filt = X_proc

        T = self.pls.transform(X_filt)
        return T

    def get_support(self) -> int:
        """Return total number of components used (predictive + orthogonal)."""
        if self.n_predictive is None or self.n_orthogonal is None:
            raise RuntimeError(
                "OPLSFeatureSelector: fit must be called before get_support."
            )
        return self.n_predictive + self.n_orthogonal

    def fit_transform(self, X: np.ndarray, y) -> np.ndarray:
        """Fit on data then transform."""
        return self.fit(X, y).transform(X)

    def prepare(
        self,
        train_X: np.ndarray,
        train_y: Union[list, np.ndarray],
        pool_X: np.ndarray,
        test_X: np.ndarray,
    ):
        """Fit on train set and transform train/pool/test splits."""
        self.fit(train_X, train_y)
        return (
            torch.tensor(self.transform(train_X), dtype=torch.double),
            torch.tensor(self.transform(pool_X), dtype=torch.double),
            torch.tensor(self.transform(test_X), dtype=torch.double),
        )

    def _fit_pls_fallback(self, X: np.ndarray, y) -> "OPLSFeatureSelector":
        """Fallback to PLS-only when pyopls is not available."""
        y_arr = np.asarray(y, dtype=float)

        # optional scaling
        if self.scale:
            self.scaler = StandardScaler()
            X_proc = self.scaler.fit_transform(X)
        else:
            X_proc = X.copy()

        n_samples, n_feats = X_proc.shape
        max_pred = min(self.max_predictive, n_feats - 1)

        best_score = -np.inf
        best_pred = 1
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # CV search for best number of PLS components
        for n_pred in range(1, max_pred + 1):
            scores = []
            for train_idx, val_idx in kf.split(X_proc):
                Xtr = X_proc[train_idx]
                Xv = X_proc[val_idx]
                ytr = y_arr[train_idx]
                yv = y_arr[val_idx]

                # Ensure n_pred doesn't exceed training fold size
                n_pred_safe = min(n_pred, len(Xtr) - 1, Xtr.shape[1] - 1)
                if n_pred_safe < 1:
                    scores.append(-np.inf)
                    continue

                try:
                    # predictive PLS only
                    pls = PLSRegression(n_components=n_pred_safe)
                    pls.fit(Xtr, ytr)
                    yv_pred = pls.predict(Xv).ravel()

                    # compute R^2
                    ss_res = np.sum((yv - yv_pred) ** 2)
                    ss_tot = np.sum((yv - yv.mean()) ** 2)
                    if ss_tot > 0:
                        scores.append(1 - ss_res / ss_tot)
                    else:
                        scores.append(0.0)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"PLS fallback CV failed: {e}")
                    scores.append(-np.inf)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_pred = n_pred

        # final fit on full data
        self.n_predictive = best_pred
        self.n_orthogonal = 0  # No orthogonal filtering in fallback mode
        self.ortho_filter = None
        
        # Enforce minimum predictive components
        self.n_predictive = max(self.min_predictive, self.n_predictive)

        # fit predictive PLS
        self.pls = PLSRegression(n_components=best_pred)
        if self.scale and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            self.pls.fit(X_scaled, y_arr)
        else:
            self.pls.fit(X, y_arr)

        if self.logger:
            self.logger.info(
                f"OPLSFeatureSelector fallback → PLS components={best_pred}, CV_score={best_score:.4f}"
            )
        return self

    def update(
        self,
        train_X: "np.ndarray",
        train_y: "np.ndarray",
        pool_X: "np.ndarray",
        test_X: "np.ndarray",
    ):
        """
        Update for BO loop: apply OPLS transformation to new data.
        """
        if self.n_predictive is None:
            raise RuntimeError("OPLS must be fitted before update")

        # Transform the new data using the already-fitted OPLS model
        train_X_transformed = self.transform(train_X)
        pool_X_transformed = self.transform(pool_X)
        test_X_transformed = self.transform(test_X)

        if self.logger:
            self.logger.info(
                f"OPLS update: transformed new data. Shapes: train={train_X_transformed.shape}, pool={pool_X_transformed.shape}, test={test_X_transformed.shape}"
            )

        return (
            torch.tensor(train_X_transformed, dtype=torch.double),
            torch.tensor(pool_X_transformed, dtype=torch.double),
            torch.tensor(test_X_transformed, dtype=torch.double),
        )
