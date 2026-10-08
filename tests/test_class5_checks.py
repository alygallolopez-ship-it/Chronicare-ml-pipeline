"""Checks taken from Class 5: fingerprint, duplicates, and the 'noise' trick."""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data, split
from src.fingerprint import data_fingerprint

EXPECTED_SHA256 = "c4a9350c40bbd585e23bec46f317f2eef99d434f4138f7ac6450c4eb01ef1744"


def test_dataset_fingerprint_unchanged():
    X, y = load_data()
    assert data_fingerprint(X, y) == EXPECTED_SHA256


def test_no_duplicate_patients_between_train_and_test():
    X, y = load_data()
    X_tr, X_te, _, _ = split(X, y, seed=0)
    overlap = X_te.merge(X_tr.drop_duplicates(), how="inner")
    assert len(overlap) == 0     # NB3 case: 29 % of test rows were copies of train


def test_noise_trick_pipeline_stays_near_chance_but_leaky_selection_does_not():
    """100 fake patients, 5000 random columns, random labels (true accuracy = 50 %)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5000))
    y = rng.integers(0, 2, size=100)

    # WRONG: select the best columns using ALL rows, then evaluate
    X_sel = SelectKBest(f_classif, k=10).fit_transform(X, y)
    leaky = cross_val_score(LogisticRegression(), X_sel, y, cv=5).mean()

    # RIGHT: selection inside the Pipeline, refit in every training fold
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=10)),
                     ("clf", LogisticRegression())])
    honest = cross_val_score(pipe, X, y, cv=5).mean()

    assert leaky > 0.75          # looks great, but it is an illusion
    assert 0.35 < honest < 0.65  # coin-flip, as it must be
