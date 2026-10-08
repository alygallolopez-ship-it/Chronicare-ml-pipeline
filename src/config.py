"""Single source of truth for every experimental setting."""

SEEDS = (0, 1, 2, 3, 4)          # the 5 seeds logged in MLflow
TEST_SIZE = 0.20                 # held-out test fraction (split depends on seed)
CV_FOLDS = 5                     # inner CV, applied to the training split only
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "chronicare-ml-pipeline"

# Hyper-parameter grids (keys are Pipeline step names: "model__<param>")
GRIDS = {
    "dummy": {},
    "ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, None],
        "model__min_samples_leaf": [1, 5],
    },
}
