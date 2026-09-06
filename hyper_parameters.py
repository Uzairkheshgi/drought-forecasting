"""Optuna search for the TWS model, scored the way the submission is: time-based
split, TWS_t masked in whole-month runs like Test.csv, recursive prediction.

Helpers mirror cell 20 of eda/StarterNotebook.ipynb — change both together.
"""

import os

# Read by OpenMP when sklearn loads, so it has to come before the import.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import gc
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

TRAIN_PATH = Path(__file__).resolve().parent / "data" / "Train.csv"

N_TRIALS = 15
FIT_FRAC = 0.2          # the model saturates far below this; keeps trials at ~4s
TWS_MASK_RATE = 0.665
TEST_RUN_LENGTHS = [1, 3, 4, 7, 1, 2]

FEATURES = ["TWS_t", "month_sin", "month_cos", "SPEI_01_t", "SPEI_03_t",
            "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]

NOTEBOOK_DEFAULTS = dict(learning_rate=0.05, max_iter=300, max_depth=8,
                         min_samples_leaf=50, l2_regularization=1.0)


def mask_tws(df, rate=TWS_MASK_RATE, seed=0):
    """Random per-row masking. For fitting only — scoring uses mask_tws_like_test."""
    out = df.copy().reset_index(drop=True)
    masked = np.random.RandomState(seed).rand(len(out)) < rate
    out.loc[masked, "TWS_t"] = np.nan
    return out


def augment_with_masking(df, rate=TWS_MASK_RATE, seed=0):
    """Original rows plus a masked copy, so the NaN branch gets trained."""
    return pd.concat([df, mask_tws(df, rate, seed)], ignore_index=True)


def assign_run_layout(times, run_lengths=TEST_RUN_LENGTHS):
    """Chop months into test-shaped runs, never spanning a calendar gap."""
    times = pd.DatetimeIndex(np.sort(np.asarray(times)))
    midx = times.year * 12 + times.month

    blocks, start = [], 0
    for i in range(1, len(times) + 1):
        if i == len(times) or midx[i] != midx[i - 1] + 1:
            blocks.append((start, i))
            start = i

    run_id = np.empty(len(times), dtype=np.int32)
    gap = np.empty(len(times), dtype=np.int32)
    r = 0
    for s, e in blocks:
        i = s
        while i < e:
            n = min(run_lengths[r % len(run_lengths)], e - i)
            run_id[i:i + n], gap[i:i + n] = r, np.arange(n)
            i += n
            r += 1

    return pd.DataFrame({"time": times, "run_id": run_id, "gap": gap})


def mask_tws_like_test(df, run_lengths=TEST_RUN_LENGTHS):
    """Mask TWS_t whole months at a time, the way Test.csv does."""
    out = df.merge(assign_run_layout(df["time"].unique(), run_lengths),
                   on="time", how="left")
    out.loc[(out["gap"] > 0).to_numpy(), "TWS_t"] = np.nan
    return out


def add_cell_id(df):
    out = df.copy()
    out["cell"] = out["lat"].astype(str) + "_" + out["lon"].astype(str)
    return out


def find_runs(df):
    """Recover the run layout from the masking pattern in the data itself."""
    observed = df.groupby("time")["TWS_t"].apply(lambda s: s.notna().mean() > 0.5)
    months = pd.DatetimeIndex(observed.index)
    midx = months.year * 12 + months.month

    runs, cur = [], []
    for i, m in enumerate(months):
        if observed.iloc[i] or not cur or midx[i] != midx[i - 1] + 1:
            if cur:
                runs.append(cur)
            cur = [m]
        else:
            cur.append(m)
    if cur:
        runs.append(cur)
    return runs


def predict_recursive(model, df, feature_names):
    """Forecast each run month by month, feeding every prediction forward."""
    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    tws_pos = feature_names.index("TWS_t")
    groups = df.groupby("time").groups

    for run in find_runs(df):
        carried = anchor = None
        for g, month in enumerate(run):
            idx = groups[month]
            block = df.loc[idx]
            # copy=True: pandas returns a read-only view under copy-on-write.
            X = block[feature_names].to_numpy(dtype=np.float32, copy=True)

            if g == 0:
                anchor = dict(zip(block["cell"], block["TWS_t"]))
            else:
                filled = block["cell"].map(carried).fillna(block["cell"].map(anchor))
                X[:, tws_pos] = filled.to_numpy(dtype=np.float32)

            pred = model.predict(X)
            out.loc[idx] = pred
            carried = dict(zip(block["cell"], pred))

    return out.to_numpy()


def build(**params):
    return HistGradientBoostingRegressor(loss="squared_error", random_state=42, **params)


def recursive_rmse(model):
    pred = predict_recursive(model, val_masked, FEATURES)
    return float(np.sqrt(mean_squared_error(y_val, pred))), pred


train = pd.read_csv(TRAIN_PATH, parse_dates=["time"]).rename(columns={"target": "Target"})

# Time-based split: a random one would put neighbouring months of the same cell on
# both sides and leak.
months = np.sort(train["time"].unique())
split = int(len(months) * 0.8)
fit_aug = augment_with_masking(train[train["time"].isin(months[:split])])
val_masked = add_cell_id(mask_tws_like_test(
    train[train["time"].isin(months[split:])].reset_index(drop=True)))

X_fit_full = fit_aug[FEATURES].to_numpy(dtype=np.float32)
y_fit_full = fit_aug["Target"].to_numpy(dtype=np.float32)
y_val = val_masked["Target"].to_numpy(dtype=np.float32)

sub = np.random.RandomState(42).choice(
    len(X_fit_full), int(FIT_FRAC * len(X_fit_full)), replace=False)
X_fit, y_fit = X_fit_full[sub], y_fit_full[sub]

del train, fit_aug          # ~2 GB of DataFrames; the float32 arrays are ~124 MB
gc.collect()

print(f"fit {X_fit.shape} of {len(X_fit_full):,} | val {len(val_masked):,}")

baseline = build(**NOTEBOOK_DEFAULTS)
baseline.fit(X_fit_full, y_fit_full)
baseline_rmse, baseline_pred = recursive_rmse(baseline)
print(f"notebook defaults: RMSE {baseline_rmse:.4f}  R2 {r2_score(y_val, baseline_pred):.4f}")


def objective(trial):
    model = build(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 500),
        # Capped at 12: an overconfident 1-step model compounds its own error at
        # every recursive step.
        max_depth=trial.suggest_int("max_depth", 4, 12),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 200),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
    )
    model.fit(X_fit, y_fit)
    return recursive_rmse(model)[0]


study = optuna.create_study(direction="minimize")
study.enqueue_trial(NOTEBOOK_DEFAULTS)
study.optimize(objective, n_trials=N_TRIALS)

# Trials ran on a subsample, so re-check the winner at full scale.
best = build(**study.best_params)
best.fit(X_fit_full, y_fit_full)
best_rmse, best_pred = recursive_rmse(best)

print(f"\nbest params: {study.best_params}")
print(f"best  RMSE {best_rmse:.4f}  R2 {r2_score(y_val, best_pred):.4f}")
print(f"defaults   RMSE {baseline_rmse:.4f}  R2 {r2_score(y_val, baseline_pred):.4f}")
print(f"gain       {baseline_rmse - best_rmse:+.4f} RMSE")
