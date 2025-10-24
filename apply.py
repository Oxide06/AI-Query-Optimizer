import json
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

def safe_makedirs(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

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
    return {
        'op_depth': float(nr.get('op_depth', 0)),
        'plan_rows_estimate': float(nr.get('plan_rows_estimate') or 0.0),
        'table_rows': 0.0,
        'num_predicates': 1.0 if nr.get('filters') else 0.0,
        'ndv_ratio': 1.0,
        'null_frac': 0.0
    }

def apply_model_to_plan(plan_json_path: str, model_path: str, out_path: str):
    with open(plan_json_path, 'r') as f:
        j = json.load(f)
    explain = j[0] if isinstance(j, list) else j
    plan = explain.get('Plan', explain)
    nodes = []
    walk_plan_and_collect_nodes(plan, nodes)
    FEATURE_COLS = ['op_depth', 'plan_rows_estimate', 'table_rows', 'num_predicates', 'ndv_ratio', 'null_frac']
    fvs = [build_feature_vector_for_node(nr) for nr in nodes]
    df_feat = pd.DataFrame(fvs)
    X = df_feat[FEATURE_COLS].fillna(0.0).astype(float)
    bst = xgb.Booster()
    bst.load_model(model_path)
    dmatrix = xgb.DMatrix(X)
    pred_log10 = bst.predict(dmatrix)
    pred_rows = np.power(10.0, pred_log10) - 1.0
    for nr, pred in zip(nodes, pred_rows):
        nr['node_obj']['LCE_Predicted_Rows'] = float(pred)
    safe_makedirs(out_path)
    with open(out_path, 'w') as f:
        json.dump(j, f, indent=2)
    print(f"Wrote adjusted plan JSON with LCE predictions to {out_path}")
