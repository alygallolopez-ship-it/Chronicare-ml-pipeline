# chronicare-ml-pipeline

Reproducible, leakage-free machine-learning pipeline (Entregable #5, Seminario Doctoral, FISI-UNMSM).
Author: Aly Gallo. Context: ChroniCare-AI doctoral thesis (Biological State Index / Adaptive Capacity).

## What this repo demonstrates
1. **No data leakage** – imputer, scaler and model live inside one scikit-learn `Pipeline`; `GridSearchCV` wraps the whole Pipeline; the train/test split happens *before* anything is fitted (`src/pipeline.py`, `src/data.py`). Nine automated tests check this (`tests/`), including the Class-5 checks: dataset SHA-256 fingerprint, no duplicate patients between train and test, and the "noise trick" (100 fake patients × 5000 random columns: leaky feature selection gives a fake high accuracy, selection inside the Pipeline stays at chance).
2. **5 seeds in MLflow, mean ± SD** – seeds `0–4` (each seed changes the train/test split, the CV folds and the model's RNG). One parent MLflow run per model, one nested child run per seed; parent logs `<metric>_mean` and `<metric>_sd` (sample SD, ddof=1).
3. **Stranger test** – pinned `requirements.txt` + fixed seeds + single-threaded training → the numbers below are reproduced exactly.

## Honest scope note
The thesis data (hospital EHR) is not available yet. This pipeline uses scikit-learn's **bundled Diabetes dataset** (Efron et al., 2004, *Annals of Statistics* 32(2):407–499; 442 patients, 10 baseline variables, target = disease progression after one year) as a **public stand-in**. The results below demonstrate the *methodology* (leakage control, multi-seed reporting, reproducibility) and are **not thesis results**. When the real data arrives, only `src/data.py` changes.

## Reproduce (stranger test)
Tested with Python 3.13 (Linux).
```bash
git clone https://github.com/alygallolopez-ship-it/Chronicare-ml-pipeline.git chronicare-ml-pipeline
cd chronicare-ml-pipeline
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                      # 9 tests (~30 s)
PYTHONHASHSEED=0 python -m src.run_experiment   # ~1–2 min; prints the table below
# same seed twice must give identical output (Class 5 test):
PYTHONHASHSEED=0 python -m src.run_experiment --models ridge --seeds 42 --out-dir /tmp/a
PYTHONHASHSEED=0 python -m src.run_experiment --models ridge --seeds 42 --out-dir /tmp/b
diff /tmp/a/per_seed_metrics.csv /tmp/b/per_seed_metrics.csv && echo IDENTICAL
mlflow ui --backend-store-uri sqlite:///mlflow.db     # optional: browse runs at http://127.0.0.1:5000
```
Outputs: `results/summary.md`, `results/summary.json`, `results/per_seed_metrics.csv` (and MLflow runs in `mlflow.db`, git-ignored).

## Expected result (held-out test set, 20 %, seeds 0–4, mean ± SD)
| model | RMSE (mean ± SD) | MAE (mean ± SD) | R² (mean ± SD) |
|---|---|---|---|
| dummy | 73.507 ± 1.169 | 62.146 ± 1.840 | -0.006 ± 0.005 |
| ridge | 55.798 ± 1.758 | 44.221 ± 1.727 | 0.419 ± 0.052 |
| random_forest | 58.483 ± 2.209 | 46.143 ± 1.645 | 0.361 ± 0.062 |

Ridge beats the mean-predicting baseline by ~18 RMSE points; the Random Forest does not beat Ridge on this small linear-ish dataset. The SD reflects variability across 5 random splits, and the test set is never used for tuning, so the estimate is not optimistically biased by model selection. It is still a small-n (5) estimate; do not read the SD as a confidence interval.

## Limits (Class 5: "reproducible is not valid")
Reproducing the same number does not make it clinically valid. Exact identity is only promised on CPU with the pinned versions (Python 3.13, Linux); other platforms/BLAS builds may differ in the last decimals. The dataset SHA-256 fingerprint (`c4a9350c…ef1744`) is logged in every MLflow parent run together with the Python version and the git commit.

## Design details
| Item | Choice |
|---|---|
| Pipeline | `SimpleImputer(median)` → `StandardScaler` → model |
| Tuning | `GridSearchCV`, 5-fold shuffled `KFold`, scoring = RMSE, **training split only** |
| Test set | touched once, for final prediction |
| Models | Dummy (mean), Ridge, Random Forest |
| Determinism | seeds in split, CV and models; `n_jobs=1`; pinned dependencies |
| Config | `src/config.py` |

## Layout
```
src/config.py          seeds, grids, MLflow settings
src/data.py            data loading + split-first
src/pipeline.py        leakage-free Pipeline + GridSearchCV
src/run_experiment.py  5-seed MLflow runner, mean ± SD
tests/                 anti-leakage and reproducibility tests
results/               exported tables from the reference run
```
