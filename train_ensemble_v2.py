#!/usr/bin/env python3
"""
train.py - FIXED VERSION with correct XGBoost parameters
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error

DATA_FILE = "data/processed/lce_training_data_fixed.csv"
MODEL_DIR = "models_improved"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading training data...")
df = pd.read_csv(DATA_FILE)

# Remove rows with invalid actual rows
df = df[df['actual_rows'] > 0]

# Create target: log of actual rows (what we want to predict)
y = np.log10(df["actual_rows"])

# CAREFUL FEATURE SELECTION - remove anything that leaks actual row information
feature_columns = [
    "node_type", "parent_node", "join_type", "relation_name", "alias",
    "filter", "hash_cond", "index_cond",
    "startup_cost", "total_cost", "plan_width", "selectivity", "cost_ratio"
]

# Only include columns that exist in the dataframe
feature_columns = [col for col in feature_columns if col in df.columns]
X = df[feature_columns].copy()

print(f"Features: {feature_columns}")
print(f"Training samples: {len(X)}")

# Handle categorical encoding with better unknown value handling
cat_cols = ["node_type", "parent_node", "join_type", "relation_name", "alias",
            "filter", "hash_cond", "index_cond"]
cat_cols = [c for c in cat_cols if c in X.columns]

encoders = {}
for col in cat_cols:
    print(f"Encoding {col}...")
    le = LabelEncoder()
    
    # Handle None/missing values before encoding
    X[col] = X[col].fillna("missing").astype(str)
    
    # Replace "None" with "none" for consistency
    X[col] = X[col].replace("None", "none")
    
    # Fit encoder and transform
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    
    # Save encoder
    joblib.dump(le, f"{MODEL_DIR}/le_{col}.pkl")
    print(f"  - {len(le.classes_)} unique values")

# Scale numerical features
num_cols = ["startup_cost", "total_cost", "plan_width", "selectivity", "cost_ratio"]
num_cols = [c for c in num_cols if c in X.columns]

print(f"Scaling numerical features: {num_cols}")
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
joblib.dump(scaler, f"{MODEL_DIR}/input_scaler.pkl")

# Train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Validation set: {X_test.shape[0]:,} samples")

# Train model with CORRECT parameters
print("\nTraining XGBoost model...")
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    # Early stopping parameters go in the constructor for XGBoost
    early_stopping_rounds=50,
    eval_metric="rmse"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# Evaluate
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Convert back from log scale to actual rows
train_actual = 10 ** y_train
train_pred = 10 ** train_preds
test_actual = 10 ** y_test  
test_pred = 10 ** test_preds

# Calculate Q-error (max(pred/actual, actual/pred))
train_q_error = np.maximum(train_pred / train_actual, train_actual / train_pred)
test_q_error = np.maximum(test_pred / test_actual, test_actual / test_pred)

print("\n=== MODEL PERFORMANCE ===")
print(f"Training set:")
print(f"  Median Q-error: {np.median(train_q_error):.3f}")
print(f"  Mean Q-error: {np.mean(train_q_error):.3f}")
print(f"  90th percentile Q-error: {np.percentile(train_q_error, 90):.3f}")

print(f"\nValidation set:")
print(f"  Median Q-error: {np.median(test_q_error):.3f}")
print(f"  Mean Q-error: {np.mean(test_q_error):.3f}")
print(f"  90th percentile Q-error: {np.percentile(test_q_error, 90):.3f}")

# Analyze performance by node type
print(f"\n=== PERFORMANCE BY NODE TYPE ===")
X_test_with_meta = X_test.copy()
X_test_with_meta['actual_rows'] = test_actual
X_test_with_meta['predicted_rows'] = test_pred
X_test_with_meta['q_error'] = test_q_error

if 'node_type' in X_test_with_meta.columns:
    # Decode node types for readability
    node_type_encoder = encoders['node_type']
    X_test_with_meta['node_type_name'] = node_type_encoder.inverse_transform(X_test_with_meta['node_type'])
    
    node_performance = X_test_with_meta.groupby('node_type_name').agg({
        'q_error': ['median', 'mean', 'count']
    }).round(3)
    print(node_performance)

# Save model
joblib.dump(model, f"{MODEL_DIR}/xgb_lce.pkl")
print(f"\nModel saved to {MODEL_DIR}/")

# Feature importance
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== TOP 10 FEATURE IMPORTANCES ===")
    print(importance_df.head(10))

print("\nTraining complete!")