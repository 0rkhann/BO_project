from .base import FeatureSelector


class VanillaSelector(FeatureSelector):
    def __init__(
        self, cv=None, max_components=None, random_state=None, logger=None, **kwargs
    ):
        # no-op, vanilla selector doesn’t need any of those
        self.logger = logger

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y):
        return X

    def get_support(self):
        return None
