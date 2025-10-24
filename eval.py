import pandas as pd
import numpy as np
import xgboost as xgb

def load_features(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, engine='python', on_bad_lines='skip', quotechar='"')

def eval_lce(features_csv: str, model_path: str):
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
    y_actual = df['actual_rows'].values
    bst = xgb.Booster()
    bst.load_model(model_path)
    dmatrix = xgb.DMatrix(X)
    pred_log10 = bst.predict(dmatrix)
    pred_rows = np.power(10, pred_log10) - 1
    q_error = np.maximum(pred_rows / (y_actual + 1e-9), (y_actual + 1e-9) / pred_rows)
    print("Evaluation Summary:")
    print(f"Median q-error: {np.median(q_error):.3f}")
    print(f"90th percentile q-error: {np.percentile(q_error, 90):.3f}")
    print(f"95th percentile q-error: {np.percentile(q_error, 95):.3f}")
