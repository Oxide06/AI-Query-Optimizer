#!/usr/bin/env python3
import json
import joblib
import numpy as np
import pandas as pd
import os
import re

PLAN_FILE = "sample_plan.json"
OUTPUT_FILE = "adjusted_plan.json"
IMPROVED_MODEL_DIR = "models_improved"
ORIGINAL_MODEL_DIR = "models"

# NORMALIZATION - UPDATED to match training
def normalize_filter(s):
    if not s or s == "None": 
        return "none"
    s = re.sub(r'[a-zA-Z_]+\.', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace('(', '').replace(')', '')
    s = s.lower()
    # Generalize numeric values like in training
    s = re.sub(r'\b\d+\b', '#', s)
    return s

def normalize_cond(s):
    if not s or s == "None": 
        return "none"
    cols = re.findall(r'([a-zA-Z_]+)\.([a-zA-Z_]+)', s)
    if len(cols) == 2:
        return f"{cols[0][1]}={cols[1][1]}".lower()
    return "none"

# FLATTEN PLAN - COMPATIBLE with both old and new models
def flatten_plan(node, parent_type="ROOT", parent_rows=None, out=None):
    if out is None: out = []
    plan_rows = node.get("Plan Rows", 1)
    selectivity = min(plan_rows / parent_rows, 1.0) if parent_rows and parent_rows > 0 else 1.0
    cost_ratio = node.get("Total Cost", 0) / (node.get("Startup Cost", 1) or 1)

    row = {
        "node_type": node.get("Node Type", "Unknown"),
        "parent_node": parent_type,
        "join_type": node.get("Join Type", "None"),
        "relation_name": node.get("Relation Name", "None"),
        "alias": node.get("Alias", "None"),
        "plan_rows": float(plan_rows),  # Keep for old model compatibility
        "startup_cost": float(node.get("Startup Cost", 0)),
        "total_cost": float(node.get("Total Cost", 0)),
        "plan_width": float(node.get("Plan Width", 0)),
        "selectivity": selectivity,
        "cost_ratio": cost_ratio,
        "log_plan_rows": np.log10(max(plan_rows, 1)),  # Keep for old model compatibility
        "filter": normalize_filter(node.get("Filter", "")),
        "hash_cond": normalize_cond(node.get("Hash Cond", "")),
        "index_cond": normalize_cond(node.get("Index Cond", "")),
    }
    out.append(row)
    for sub in node.get("Plans", []):
        flatten_plan(sub, node.get("Node Type", "Unknown"), plan_rows, out)
    return out

# EXTRACT ACTUAL ROWS FROM ORIGINAL PLAN
def extract_actual_rows(node, out=None):
    if out is None: out = []
    out.append(node.get("Actual Rows", 0))
    for sub in node.get("Plans", []):
        extract_actual_rows(sub, out)
    return out

# FEATURE SELECTION based on model type
def get_features_for_model(df, model_type="improved"):
    """Select appropriate features based on model type"""
    if model_type == "improved":
        # New model features (no data leakage)
        num_cols = ["startup_cost", "total_cost", "plan_width", "selectivity", "cost_ratio"]
        cat_cols = ["node_type", "parent_node", "join_type", "relation_name", "alias",
                   "filter", "hash_cond", "index_cond"]
    else:
        # Old model features (with plan_rows)
        num_cols = ["plan_rows", "startup_cost", "total_cost", "plan_width",
                   "selectivity", "cost_ratio", "log_plan_rows"]
        cat_cols = ["node_type", "parent_node", "join_type", "relation_name", "alias",
                   "filter", "hash_cond", "index_cond"]
    
    # Only include columns that exist in dataframe
    num_cols = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    return num_cols, cat_cols

# CREATE COMPATIBLE FEATURE SET
def create_compatible_features(df, expected_features):
    """Create a feature matrix with all expected features, filling missing ones with 0"""
    result = pd.DataFrame()
    for feature in expected_features:
        if feature in df.columns:
            result[feature] = df[feature]
        else:
            # Fill missing features with 0 (or appropriate default)
            if feature in ["plan_rows", "log_plan_rows"]:
                # For these specific features, calculate them
                if feature == "plan_rows" and "plan_rows" not in df.columns:
                    result["plan_rows"] = df.get("Plan Rows", 1)
                elif feature == "log_plan_rows" and "log_plan_rows" not in df.columns:
                    result["log_plan_rows"] = np.log10(df.get("plan_rows", 1))
            else:
                result[feature] = 0
    return result

# MAIN
def main():
    if not os.path.exists(PLAN_FILE):
        print("sample_plan.json not found!")
        return

    model_type = "improved"
    fallback_used = False
    MODEL_DIR = IMPROVED_MODEL_DIR  # Initialize MODEL_DIR
    
    # Try to load improved model first
    try:
        print("Loading improved model from models_improved/...")
        model = joblib.load(f"{MODEL_DIR}/xgb_lce.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/input_scaler.pkl")
        encoders = {}
        for col in ["node_type", "parent_node", "join_type", "relation_name", "alias",
                    "filter", "hash_cond", "index_cond"]:
            p = f"{MODEL_DIR}/le_{col}.pkl"
            if os.path.exists(p):
                encoders[col] = joblib.load(p)
        print("Loaded improved model successfully!")
        
    except Exception as e:
        print(f"Error loading improved model: {e}")
        print("Falling back to original models/ directory...")
        MODEL_DIR = ORIGINAL_MODEL_DIR  # Properly reassign
        model_type = "original"
        fallback_used = True
        
        try:
            model = joblib.load(f"{MODEL_DIR}/xgb_lce.pkl")
            scaler = joblib.load(f"{MODEL_DIR}/input_scaler.pkl")
            encoders = {}
            for col in ["node_type", "parent_node", "join_type", "relation_name", "alias",
                        "filter", "hash_cond", "index_cond"]:
                p = f"{MODEL_DIR}/le_{col}.pkl"
                if os.path.exists(p):
                    encoders[col] = joblib.load(p)
            print("Loaded original model successfully!")
        except Exception as e2:
            print(f"Error loading original model: {e2}")
            return

    # LOAD ORIGINAL PLAN
    with open(PLAN_FILE) as f:
        original_plan = json.load(f)
    root_node = original_plan[0]["Plan"]

    # EXTRACT ACTUAL ROWS BEFORE ANY MODIFICATION
    actual_rows = extract_actual_rows(root_node)

    # FLATTEN FOR PREDICTION
    nodes = flatten_plan(root_node)
    df = pd.DataFrame(nodes)

    print(f"Processing {len(df)} nodes with {model_type} model...")

    # ENCODE - with proper handling of "None" values
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
            # Replace "None" with "none" to match training
            df[col] = df[col].replace("None", "none")
            # Encode with proper unknown value handling
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # GET APPROPRIATE FEATURES for the model type
    num_cols, cat_cols = get_features_for_model(df, model_type)
    
    print(f"Using numerical features: {num_cols}")
    print(f"Using categorical features: {cat_cols}")

    # CREATE COMPATIBLE FEATURE SET
    try:
        # Get all expected features from the scaler
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else num_cols
        
        # Create a feature matrix with all expected features
        X_compatible = create_compatible_features(df, expected_features)
        
        # Ensure the order matches what the scaler expects
        X_compatible = X_compatible[expected_features]
        
        # Scale the features
        X_num = scaler.transform(X_compatible)
        
    except Exception as e:
        print(f"Feature processing error: {e}")
        print("Attempting alternative approach...")
        
        # Alternative: Use only available features that match
        available_features = [f for f in expected_features if f in df.columns]
        print(f"Using available features: {available_features}")
        
        if not available_features:
            print("No compatible features found!")
            return
            
        X_compatible = df[available_features]
        X_num = scaler.transform(X_compatible)

    X_cat = df[cat_cols].values
    X = np.hstack([X_cat, X_num])

    print(f"Feature matrix shape: {X.shape}")

    # PREDICT
    preds_log = model.predict(X)
    ai_rows = 10 ** preds_log

    # Add some basic sanity checks
    def cap_predictions(predictions, min_rows=0.1, max_rows=100000):
        """Cap predictions to reasonable ranges"""
        return np.clip(predictions, min_rows, max_rows)
    
    ai_rows = cap_predictions(ai_rows)

    # INJECT INTO A COPY
    plan_copy = json.loads(json.dumps(original_plan))  # deep copy
    def inject(node, i=[0]):
        node["AI_Estimated_Rows"] = float(ai_rows[i[0]])
        i[0] += 1
        for sub in node.get("Plans", []): 
            inject(sub, i)
        return node
    
    plan_copy[0]["Plan"] = inject(plan_copy[0]["Plan"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(plan_copy, f, indent=4)

    # IMPROVED DEBUG OUTPUT
    print(f"\n=== {'IMPROVED' if model_type == 'improved' else 'ORIGINAL'} MODEL RESULTS ===")
    print(f"{'Node':4} {'Type':12} {'PG Estimate':>12} {'AI Estimate':>12} {'Actual':>12} {'Q-PG':>8} {'Q-AI':>8}")
    print("-" * 80)
    
    total_pg_error = 0
    total_ai_error = 0
    ai_wins = 0
    pg_wins = 0
    valid_nodes = 0
    
    for i in range(len(ai_rows)):
        pg = df["plan_rows"].iloc[i] if "plan_rows" in df.columns else 0
        ai = ai_rows[i]
        act = actual_rows[i]
        
        if act > 0:
            q_pg = max(pg/act, act/pg)
            q_ai = max(ai/act, act/ai)
            
            total_pg_error += q_pg
            total_ai_error += q_ai
            valid_nodes += 1
            
            if q_ai < q_pg:
                ai_wins += 1
                winner = "AI"
            elif q_ai > q_pg:
                pg_wins += 1
                winner = "PG"
            else:
                winner = "TIE"
        else:
            q_pg = 1.0
            q_ai = 1.0
            winner = "N/A"
        
        node_type = df["node_type"].iloc[i] if i < len(df) else "Unknown"
        print(f"{i:4} {node_type:12} {pg:12.0f} {ai:12.0f} {act:12.0f} {q_pg:8.2f} {q_ai:8.2f} {winner}")

    # SUMMARY STATISTICS
    print("-" * 80)
    if valid_nodes > 0:
        avg_pg_error = total_pg_error / valid_nodes
        avg_ai_error = total_ai_error / valid_nodes
        improvement = ((avg_pg_error - avg_ai_error) / avg_pg_error * 100) if avg_pg_error > 0 else 0
        
        print(f"SUMMARY:")
        print(f"  Model Type:            {model_type.upper()}")
        if fallback_used:
            print(f"Using fallback model")
        print(f"  Average PG Error Ratio:  {avg_pg_error:.2f}")
        print(f"  Average AI Error Ratio:  {avg_ai_error:.2f}")
        print(f"  Improvement:            {improvement:+.1f}%")
        print(f"  AI Wins:                {ai_wins} nodes")
        print(f"  PostgreSQL Wins:        {pg_wins} nodes")
        
        if ai_wins > pg_wins:
            print(f"AI model is performing better!")
        else:
            print(f"PostgreSQL estimates are better")
    
    print(f"\nAdjusted plan saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()