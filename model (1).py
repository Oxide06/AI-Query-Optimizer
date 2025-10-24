import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

def safe_makedirs(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def load_features(csv_path: str) -> pd.DataFrame:
    # Use Python engine and skip malformed lines
    return pd.read_csv(csv_path, engine='python', on_bad_lines='skip', quotechar='"')

def train_lce(features_csv: str, model_out: str, test_size=0.15, random_state=42):
    df = load_features(features_csv)
    # Convert categorical columns
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

def walk_plan_and_collect_nodes(plan, nodes, parent_id=None, depth=0):
    node = plan
    node_id = id(node)
    node_record = {
        'node_obj': node,
        'node_id': node_id,
        'op_depth': depth,
        'plan_rows_estimate': node.get('Plan Rows', node.get('Plan Rows Estimate', 0)),
        'relation_name': node.get('Relation Name', None),
        'filters': node.get('Filter', None),
        'node_type': node.get('Node Type', None)
    }
    nodes.append(node_record)
    for child in node.get('Plans', []):
        walk_plan_and_collect_nodes(child, nodes, parent_id=node_id, depth=depth+1)

def build_feature_vector_for_node(node_record):
    nr = node_record
    fv = {
        'op_depth': float(nr.get('op_depth', 0)),
        'plan_rows_estimate': float(nr.get('plan_rows_estimate') or 0.0),
        'table_rows': 0.0,
        'num_predicates': 1.0 if nr.get('filters') else 0.0,
        'ndv_ratio': 1.0,
        'null_frac': 0.0
    }
    return fv

def apply_model_to_plan(plan_json_path: str, model_path: str, out_path: str):
    with open(plan_json_path, 'r') as f:
        j = json.load(f)
    explain = j[0] if isinstance(j, list) else j
    plan = explain.get('Plan', explain)
    nodes = []
    walk_plan_and_collect_nodes(plan, nodes)
    FEATURE_COLS = [
        'op_depth', 'plan_rows_estimate', 'table_rows',
        'num_predicates', 'ndv_ratio', 'null_frac'
    ]
    fvs = [build_feature_vector_for_node(nr) for nr in nodes]
    df_feat = pd.DataFrame(fvs)
    X = df_feat[FEATURE_COLS].fillna(0.0).astype(float)
    bst = xgb.Booster()
    bst.load_model(model_path)
    dmatrix = xgb.DMatrix(X)
    pred_log10 = bst.predict(dmatrix)
    pred_rows = np.power(10.0, pred_log10) - 1.0
    for nr, pred in zip(nodes, pred_rows):
        node_obj = nr['node_obj']
        node_obj['LCE_Predicted_Rows'] = float(pred)
    safe_makedirs(out_path)
    with open(out_path, 'w') as f:
        json.dump(j, f, indent=2)
    print(f"Wrote adjusted plan JSON with LCE predictions to {out_path}")

def main():
    parser = argparse.ArgumentParser(description='AI LCE CLI')
    sub = parser.add_subparsers(dest='cmd')
    p_train = sub.add_parser('train')
    p_train.add_argument('--features', required=True)
    p_train.add_argument('--model', required=True)
    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--features', required=True)
    p_eval.add_argument('--model', required=True)
    p_apply = sub.add_parser('apply')
    p_apply.add_argument('--plan', required=True)
    p_apply.add_argument('--model', required=True)
    p_apply.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.cmd == 'train':
        train_lce(args.features, args.model)
    elif args.cmd == 'eval':
        eval_lce(args.features, args.model)
    elif args.cmd == 'apply':
        apply_model_to_plan(args.plan, args.model, args.out)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
