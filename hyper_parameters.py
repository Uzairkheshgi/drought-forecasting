import optuna
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data") 
TRAIN_FEATURES_PATH = DATA_DIR / "Train.csv"

def normalise_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Some releases use 'sample_id' instead of 'ID' for the test file — normalise it."""
    if "sample_id" in df.columns and "ID" not in df.columns:
        out = df.copy()
        out = out.rename(columns={"sample_id": "ID"})
        return out
    return df


train_features = normalise_id_column(pd.read_csv(TRAIN_FEATURES_PATH, parse_dates=["time"]))
# 1. Isolate training features and the target variable
features = [
    'SOIL_MOISTURE_t', 'SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 
    # 'SPEI_12_t', 'TWS_t', 'month_cos', 'month_sin'
]

# Tip: To speed up the search on 2.15M rows, you can sample 10% of the data initially
# X_train = train_features[features].sample(frac=0.1, random_state=42)
# y_train = train_features['target'].loc[X_train.index]
X_train = train_features[features]
y_train = train_features['target']

def objective(trial):
    # 2. Suggest hyperparameters dynamically
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 500)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    l2_regularization = trial.suggest_float('l2_regularization', 0.0, 1.0)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 15, 63)
    
    # 3. Instantiate the regressor (natively handles NaNs in TWS_t)
    hgbr = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        l2_regularization=l2_regularization,
        max_leaf_nodes=max_leaf_nodes,
        loss='squared_error',
        random_state=42
    )
    
    # 4. Evaluate using 3-fold cross-validation
    score = cross_val_score(
        hgbr, X_train, y_train, cv=3, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1
    ).mean()
    
    return score

# 5. Create study and optimize (maximizing the negative RMSE brings it closer to 0)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15) # Keep trials low initially to save time

print(f"Best Hyperparameters: {study.best_params}")
print(f"Best CV RMSE: {np.abs(study.best_value):.4f}")