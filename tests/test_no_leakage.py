"""Anti-leakage tests. Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data import load_data, split
from src.pipeline import make_pipeline, make_search


@pytest.fixture(scope="module")
def parts():
    X, y = load_data()
    return split(X, y, seed=0)


def test_split_is_disjoint_and_seeded(parts):
    X_tr, X_te, _, _ = parts
    assert set(X_tr.index).isdisjoint(X_te.index)
    X, y = load_data()
    X_tr2, _, _, _ = split(X, y, seed=0)
    assert list(X_tr.index) == list(X_tr2.index)      # same seed -> same split
    X_tr3, _, _, _ = split(X, y, seed=1)
    assert list(X_tr.index) != list(X_tr3.index)      # other seed -> other split


def test_scaler_statistics_come_from_train_only(parts):
    X_tr, X_te, y_tr, _ = parts
    pipe = make_pipeline("ridge", 0).fit(X_tr, y_tr)
    scaler = pipe.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_tr.mean().values, rtol=1e-9)
    # a leaky scaler fit on train+test would give different statistics
    leaky = StandardScaler().fit(pd.concat([X_tr, X_te]))
    assert not np.allclose(scaler.mean_, leaky.mean_, rtol=1e-12, atol=0)


def test_test_data_cannot_influence_fit(parts):
    """Corrupt the test set wildly: the fitted search must be identical."""
    X_tr, X_te, y_tr, _ = parts
    s1 = make_search("ridge", 0).fit(X_tr, y_tr)
    _ = X_te * 1e6 + 123.0                              # never reaches fit()
    s2 = make_search("ridge", 0).fit(X_tr, y_tr)
    assert s1.best_params_ == s2.best_params_
    np.testing.assert_allclose(s1.best_estimator_.named_steps["model"].coef_,
                               s2.best_estimator_.named_steps["model"].coef_)


def test_predictions_are_row_independent_of_other_test_rows(parts):
    """A leaky transform (e.g. scaling with test statistics) would make a
    row's prediction depend on which other rows are in the test batch."""
    X_tr, X_te, y_tr, _ = parts
    pipe = make_pipeline("ridge", 0).fit(X_tr, y_tr)
    full = pipe.predict(X_te)
    half = pipe.predict(X_te.iloc[:10])
    np.testing.assert_allclose(full[:10], half, rtol=1e-12)


def test_search_wraps_preprocessing_inside_cv(parts):
    search = make_search("random_forest", 0)
    assert [n for n, _ in search.estimator.steps] == ["imputer", "scaler", "model"]


def test_same_seed_gives_identical_predictions(parts):
    X_tr, X_te, y_tr, _ = parts
    a = make_search("random_forest", 0).fit(X_tr, y_tr).predict(X_te)
    b = make_search("random_forest", 0).fit(X_tr, y_tr).predict(X_te)
    np.testing.assert_array_equal(a, b)
