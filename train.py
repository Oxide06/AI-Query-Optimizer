import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pathlib import Path

def safe_makedirs(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def load_features(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, engine='python', on_bad_lines='skip', quotechar='"')

def train_lce(features_csv: str, model_out: str, test_size=0.15, random_state=42):
    df = load_features(features_csv)
    for col in ['node_type', 'join_type', 'relation', 'parent_node', 'query_name']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    df['has_filter'] = df['filter'].notnull().astype(int) if 'filter' in df.columns else 0
    if 'filter' in df.columns:
        df = df.drop(columns=['filter'])
    FEATURE_COLS = [
        'node_type', 'relation', 'join_type', 'plan_rows',
        'total_cost', 'actual_time', 'parent_node', 'query_name', 'has_filter'
    ]
    X = df[FEATURE_COLS].fillna(0.0).astype(float)
    y = np.log10(df['actual_rows'] + 1.0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist'}
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=500, evals=evallist, early_stopping_rounds=20, verbose_eval=20)
    safe_makedirs(model_out)
    bst.save_model(model_out)
    print(f"Model saved to {model_out}. Best iteration: {bst.best_iteration}")
