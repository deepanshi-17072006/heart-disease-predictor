"""Shared preprocessing for training (train.py) and serving (app.py).

The mid-term notebook did these steps by hand, one after another. Here they are
packed into ONE scikit-learn Pipeline so that model/model.joblib contains the
complete chain and the API cannot apply the steps in a different order.

Steps (same as the notebook, all learned on the training split only):
  1. One-hot encode the nominal columns `cp` and `thal`
  2. Bin `age` into 3 ordinal groups (<=44, 45-59, 60+)
  3. log1p on columns whose training skewness is > 0.75 in absolute value
  4. RobustScaler (median / IQR)
  5. SelectKBest(f_classif, k=15)
  6. The classifier (logistic regression / random forest / gradient boosting)
"""
import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"
NOMINAL = ["cp", "thal"]
PASSTHROUGH = [c for c in FEATURES if c not in NOMINAL + ["age"]]

# Allowed range per feature (the data contract). app.py enforces the same rules
# in its Pydantic schema; train.py uses this dict to check the CSV.
FEATURE_BOUNDS = {
    "age": (29, 77), "sex": (0, 1), "cp": (0, 3), "trestbps": (80, 220),
    "chol": (100, 600), "fbs": (0, 1), "restecg": (0, 2), "thalach": (60, 220),
    "exang": (0, 1), "oldpeak": (0.0, 7.0), "slope": (0, 2), "ca": (0, 4),
    "thal": (0, 3),
}


def age_to_group(X):
    """Ordinal age group: 0 = up to 44, 1 = 45-59, 2 = 60 and above.

    Same bin edges as the notebook (29, 44, 59, 77) but never returns NaN, so
    an unexpected age cannot crash the API.
    """
    a = np.asarray(X, dtype=float)
    return np.select([a <= 44, a <= 59], [0.0, 1.0], default=2.0)


class SkewLog1p(BaseEstimator, TransformerMixin):
    """Apply log1p to columns that are skewed in the TRAINING data."""

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            s = skew(X, axis=0)  # constant columns give NaN -> never selected
        self.skewed_idx_ = np.where(np.abs(s) > self.threshold)[0]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if len(self.skewed_idx_):
            X[:, self.skewed_idx_] = np.log1p(np.clip(X[:, self.skewed_idx_], 0, None))
        return X


def build_pipeline(estimator, k_features=15):
    """Full preprocessing + model pipeline. Input: a DataFrame with FEATURES."""
    columns = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL),
        ("age_group", FunctionTransformer(age_to_group, validate=False), ["age"]),
        ("rest", "passthrough", PASSTHROUGH),
    ])
    return Pipeline([
        ("columns", columns),
        ("skew", SkewLog1p(threshold=0.75)),
        ("scale", RobustScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k_features)),
        ("clf", estimator),
    ])
