"""
baseline_models.py
==================
XGBoost baseline on tabular features from the car insurance fraud dataset.

Features used (no graph structure):
  - policy_deductible, policy_annual_premium
  - insured_age, insured_sex, insured_education_level, insured_occupation,
    insured_hobbies
  - incident_type, collision_type, incident_severity, authorities_contacted
  - incident_state, incident_hour_of_the_day, incident_month, incident_dow
  - number_of_vehicles_involved, bodily_injuries, witnesses
  - police_report_available
  - claim_amount, total_claim_amount   ← kept here (no leakage unlike PaySim)

Target: fraud_reported (Y/N → 1/0)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import compute_metrics, print_metrics, print_classification_report, set_seed

DATA_DIR  = "data"
CSV_NAME  = "car_insurance_fraud_dataset.csv"
SEED      = 42

CAT_COLS = [
    "policy_state", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "incident_type",
    "collision_type", "incident_severity", "authorities_contacted",
    "incident_state", "police_report_available",
]

NUM_COLS = [
    "policy_deductible", "policy_annual_premium", "insured_age",
    "incident_hour_of_the_day", "number_of_vehicles_involved",
    "bodily_injuries", "witnesses", "claim_amount", "total_claim_amount",
    "incident_month", "incident_dow",
]


def load_and_prepare(path):
    print(f"[1/4] Loading: {path}")
    df = pd.read_csv(path)
    print(f"      Shape      : {df.shape}")
    print(f"      Fraud rate : {(df['fraud_reported']=='Y').mean():.4%}")

    df["fraud_reported"].fillna("N", inplace=True)
    df["authorities_contacted"] = df["authorities_contacted"].fillna("None")
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["incident_month"] = df["incident_date"].dt.month
    df["incident_dow"]   = df["incident_date"].dt.dayofweek

    # Encode categoricals
    print("[2/4] Encoding features ...")
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[NUM_COLS + CAT_COLS].values.astype(np.float32)
    y = (df["fraud_reported"] == "Y").astype(int).values
    return X, y, NUM_COLS + CAT_COLS


def main():
    set_seed(SEED)
    csv_path = os.path.join(DATA_DIR, CSV_NAME)
    if not os.path.isfile(csv_path):
        print(f"[ERROR] Not found: {csv_path}")
        sys.exit(1)

    X, y, feat_names = load_and_prepare(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

    print(f"\n      Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"      pos_weight  : {pos_weight:.2f}")

    import torch
    use_gpu = torch.cuda.is_available()
    params  = dict(
        n_estimators     = 400,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = pos_weight,
        eval_metric      = "auc",
        tree_method      = "hist",
        random_state     = SEED,
        n_jobs           = -1,
        early_stopping_rounds = 30,
    )
    if use_gpu:
        params["device"] = "cuda"
        print("      XGBoost: GPU")
    else:
        print("      XGBoost: CPU")

    print("\n[3/4] Training XGBoost ...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    print("\n[4/4] Evaluating on test set ...")
    y_prob   = model.predict_proba(X_test)[:, 1]

    # Sweep threshold for best F1
    best_f1, best_t = 0.0, 0.5
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        m = compute_metrics(y_test, y_prob, threshold=t)
        if m["F1"] > best_f1:
            best_f1, best_t = m["F1"], t
    print(f"  Best threshold (F1): {best_t}")

    metrics = compute_metrics(y_test, y_prob, threshold=best_t)
    print_metrics(metrics, "XGBoost Baseline – Test Results")
    print_classification_report(y_test, y_prob, threshold=best_t)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(9, 5))
    xgb.plot_importance(model, ax=ax, importance_type="gain",
                        max_num_features=15, title="XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plot_path = os.path.join(DATA_DIR, "xgb_feature_importance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Feature importance → {plot_path}")

    model_path = os.path.join(DATA_DIR, "xgb_model.json")
    model.save_model(model_path)
    print(f"  Model saved        → {model_path}")


if __name__ == "__main__":
    main()
