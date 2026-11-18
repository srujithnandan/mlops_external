from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add the two derived features used by the training notebook."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["nozzle_bed_temperature_difference"] = (
            X_copy["nozzle_temperature"] - X_copy["bed_temperature"]
        )
        X_copy["cooling_stress_index"] = X_copy["cooling_fan_speed"] * (
            X_copy["printing_speed"] / (X_copy["ambient_temperature"] + 1e-5)
        )
        return X_copy
