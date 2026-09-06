"""Optuna search for the TWS forecasting model.

Tuned against the protocol the submission actually uses: a time-based split, TWS_t
masked in whole-month runs the way Test.csv masks it, and **recursive** prediction
through each run. Tuning against a random split, an unmasked validation set, or
NaN-when-masked all optimise for a problem the submission never faces — see
eda/StarterNotebook.ipynb.

NOTE: the helpers below mirror cell 20 of the notebook. They are duplicated rather
than imported so the notebook stays self-contained; if you change one, change both.
"""

import os

# Must precede the sklearn import: HistGradientBoostingRegressor parallelises through
# OpenMP, which reads this at load time. Leaving headroom keeps the box usable and
# avoids thrashing if anything else is running.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import gc
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_FEATURES_PATH = DATA_DIR / "Train.csv"

N_TRIALS = 15
FIT_FRAC = 1.0          # drop to e.g. 0.3 to trade accuracy for speed while exploring
TWS_MASK_RATE = 0.665   # share of test rows with TWS_t hidden

# Run lengths observed in Test.csv: only each run's first month keeps TWS_t.
TEST_RUN_LENGTHS = [1, 3, 4, 7, 1, 2]

FEATURES = [
    "TWS_t",
    "month_sin",
    "month_cos",
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
]


def normalise_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Some releases use 'sample_id' instead of 'ID' for the test file — normalise it."""
    if "sample_id" in df.columns and "ID" not in df.columns:
        return df.rename(columns={"sample_id": "ID"})
    return df


def add_cell_id(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable grid-cell key so runs can be chained location by location."""
    out = df.copy()
    out["cell"] = out["lat"].astype(str) + "_" + out["lon"].astype(str)
    return out


def mask_tws(df: pd.DataFrame, rate: float = TWS_MASK_RATE, seed: int = 0) -> pd.DataFrame:
    """Copy of `df` with TWS_t set to NaN on a random `rate` share of rows.

    Random masking is for *fitting* only — there we just need varied examples of
    "TWS_t is absent". Scoring uses mask_tws_like_test.
    """
    out = df.copy().reset_index(drop=True)
    masked = np.random.RandomState(seed).rand(len(out)) < rate
    out.loc[masked, "TWS_t"] = np.nan
    return out


def augment_with_masking(df: pd.DataFrame, rate: float = TWS_MASK_RATE,
                         seed: int = 0) -> pd.DataFrame:
    """Original rows plus a masked copy, so the NaN branch is trained as a safety net."""
    return pd.concat([df, mask_tws(df, rate, seed)], ignore_index=True)


def assign_run_layout(times, run_lengths=TEST_RUN_LENGTHS) -> pd.DataFrame:
    """Chop months into runs shaped like the test set, never spanning a calendar gap."""
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
            length = min(run_lengths[r % len(run_lengths)], e - i)
            run_id[i:i + length] = r
            gap[i:i + length] = np.arange(length)
            i += length
            r += 1

    return pd.DataFrame({"time": times, "run_id": run_id, "gap": gap})


def mask_tws_like_test(df: pd.DataFrame, run_lengths=TEST_RUN_LENGTHS) -> pd.DataFrame:
    """Mask TWS_t the way Test.csv does: whole months at a time, in runs."""
    layout = assign_run_layout(df["time"].unique(), run_lengths)
    out = df.merge(layout, on="time", how="left")  # left join preserves row order
    masked = (out["gap"] > 0).to_numpy()
    out.loc[masked, "TWS_t"] = np.nan
    out["TWS_t_masked"] = masked
    return out


def find_runs(df: pd.DataFrame) -> list:
    """Group months into runs by reading the TWS_t masking pattern itself."""
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


def predict_recursive(model, df: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Forecast through each run, feeding every month's prediction forward.

    target(cell, M) IS TWS(cell, M+1), so only a run's anchor month needs an observed
    TWS_t. Each step also picks up that month's real SPEI and soil moisture.
    """
    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    tws_pos = feature_names.index("TWS_t")
    groups = df.groupby("time").groups

    for run in find_runs(df):
        carried, anchor = None, None
        for g, month in enumerate(run):
            idx = groups[month]
            block = df.loc[idx]
            # copy=True: pandas hands back a read-only view under copy-on-write.
            X = block[feature_names].to_numpy(dtype=np.float32, copy=True)

            if g == 0:
                anchor = dict(zip(block["cell"], block["TWS_t"]))
            else:
                filled = block["cell"].map(carried)
                filled = filled.fillna(block["cell"].map(anchor))
                X[:, tws_pos] = filled.to_numpy(dtype=np.float32)

            pred = model.predict(X)
            out.loc[idx] = pred
            carried = dict(zip(block["cell"], pred))

    return out.to_numpy()


# --------------------------------------------------------------------------- data
train_features = normalise_id_column(
    pd.read_csv(TRAIN_FEATURES_PATH, parse_dates=["time"])
).rename(columns={"target": "Target"})

# Time-based split: the last 20% of months are held out. A random split would put
# neighbouring months of the same grid cell on both sides and leak.
unique_times = np.sort(train_features["time"].unique())
split_idx = int(len(unique_times) * 0.8)
fit_df = train_features[train_features["time"].isin(unique_times[:split_idx])]
val_df = train_features[train_features["time"].isin(unique_times[split_idx:])].reset_index(drop=True)

fit_aug = augment_with_masking(fit_df, seed=0)
if FIT_FRAC < 1.0:
    fit_aug = fit_aug.sample(frac=FIT_FRAC, random_state=42)

X_fit = fit_aug[FEATURES].to_numpy(dtype=np.float32)
y_fit = fit_aug["Target"].to_numpy(dtype=np.float32)

# Validation stays a DataFrame — predict_recursive needs time/cell to chain runs.
val_masked = add_cell_id(mask_tws_like_test(val_df))
y_val = val_masked["Target"].to_numpy(dtype=np.float32)
val_gap = val_masked["gap"].to_numpy()

# The DataFrames behind X_fit are ~2 GB and are not needed again.
del train_features, fit_df, fit_aug
gc.collect()

print(f"fit rows: {X_fit.shape}   val rows: {len(val_masked):,}")
print(f"val runs (month counts): {[len(r) for r in find_runs(val_masked)]}")
print(f"masked share: {val_masked['TWS_t_masked'].mean():.1%}  (test is 66.5%)")


def evaluate(model) -> float:
    """Recursive-RMSE on the validation runs — the quantity the submission is scored on."""
    return float(np.sqrt(mean_squared_error(y_val, predict_recursive(model, val_masked, FEATURES))))


def build(**params) -> HistGradientBoostingRegressor:
    # No imputer: HistGradientBoostingRegressor routes NaN down a learned branch.
    return HistGradientBoostingRegressor(loss="squared_error", random_state=42, **params)


# ---------------------------------------------------------------------- baseline
NOTEBOOK_DEFAULTS = dict(learning_rate=0.05, max_iter=300, max_depth=8,
                         min_samples_leaf=50, l2_regularization=1.0)
baseline_model = build(**NOTEBOOK_DEFAULTS)
baseline_model.fit(X_fit, y_fit)
baseline_pred = predict_recursive(baseline_model, val_masked, FEATURES)
baseline_rmse = float(np.sqrt(mean_squared_error(y_val, baseline_pred)))
print(f"\nnotebook defaults: recursive RMSE {baseline_rmse:.4f}  "
      f"R2 {r2_score(y_val, baseline_pred):.4f}\n")


def objective(trial):
    model = build(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 500),
        # Capped at 12: deep trees are slow here and chain badly — an overconfident
        # 1-step model compounds its own error at every recursive step.
        max_depth=trial.suggest_int("max_depth", 4, 12),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 200),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
    )
    model.fit(X_fit, y_fit)
    return evaluate(model)


study = optuna.create_study(direction="minimize")
study.enqueue_trial(NOTEBOOK_DEFAULTS)  # start from what the notebook already uses
study.optimize(objective, n_trials=N_TRIALS)

print(f"\nBest hyperparameters: {study.best_params}")
print(f"Best recursive RMSE:  {study.best_value:.4f}")
print(f"Notebook defaults:    {baseline_rmse:.4f}")
gain = baseline_rmse - study.best_value
print(f"Improvement:          {gain:+.4f} RMSE"
      f"{'  (not worth changing)' if gain < 0.002 else ''}")
