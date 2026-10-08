"""Leakage-free modelling pipeline.

Every data-dependent step (imputer, scaler, model) lives INSIDE a single
sklearn Pipeline, and hyper-parameter search (GridSearchCV) wraps that whole
Pipeline. Hence in every inner CV fold, and in the final refit, the imputer
and scaler statistics are learned from training rows only.
"""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CV_FOLDS, GRIDS


def make_model(name: str, seed: int):
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge(random_state=seed)
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1)
    raise ValueError(f"unknown model: {name}")


def make_pipeline(name: str, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", make_model(name, seed)),
    ])


def make_search(name: str, seed: int) -> GridSearchCV:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    return GridSearchCV(
        make_pipeline(name, seed),
        param_grid=GRIDS[name] or {},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        n_jobs=1,
    )
