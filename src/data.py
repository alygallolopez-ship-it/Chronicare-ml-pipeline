"""Data loading and splitting.

Dataset: scikit-learn's bundled Diabetes data (Efron, Hastie, Johnstone &
Tibshirani, 2004, Annals of Statistics 32(2):407-499): 442 patients, 10
baseline variables, target = disease progression one year after baseline.
It ships inside scikit-learn, so no download is needed. It is a PUBLIC
STAND-IN for the (unavailable) hospital EHR of the thesis; see README.
"""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from .config import TEST_SIZE


def load_data():
    bunch = load_diabetes(as_frame=True)
    return bunch.data.copy(), bunch.target.copy()


def split(X: pd.DataFrame, y: pd.Series, seed: int):
    """Split FIRST, before any preprocessing. The test set is never touched
    by imputation, scaling or hyper-parameter search."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
