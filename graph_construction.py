"""
graph_construction.py
=====================
Loads the Car Insurance Fraud dataset and constructs a bipartite graph:

    Graph design:
    - Nodes  : insured persons (policy_id) + incident cities (incident_city)
    - Edges  : one directed edge per claim  (policy → city)
    - Edge features  : [claim_amount_log, total_claim_log, incident_type_enc,
                        incident_severity_enc, hour_of_day_norm,
                        num_vehicles, bodily_injuries, witnesses,
                        police_report, authorities_enc, deductible_log,
                        annual_premium_log]
    - Edge label     : fraud_reported  (1 = Y, 0 = N)
    - Node features  : per-node aggregated statistics (8 features)

Why a bipartite policy→city graph?
    Fraudsters often cluster around the same incident city or share
    the same network of cities. Linking policies through shared cities
    lets GraphSAGE propagate fraud signals across the graph.

Dataset: Car Insurance Fraud Dataset (30,000 claims)
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

DATA_DIR  = "data"
CSV_NAME  = "car_insurance_fraud_dataset.csv"
OUT_NAME  = "insurance_graph.pt"
SEED      = 42


# ──────────────────────────────────────────────
# 1. Load & Clean
# ──────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"      Raw shape  : {df.shape}")
    print(f"      Fraud rate : {(df['fraud_reported']=='Y').mean():.4%}")

    # Fill missing authorities_contacted with 'None'
    df["authorities_contacted"] = df["authorities_contacted"].fillna("None")

    # Convert target to binary int
    df["label"] = (df["fraud_reported"] == "Y").astype(np.float32)

    # Parse date → extract month and day-of-week as extra features
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["incident_month"] = df["incident_date"].dt.month.astype(np.float32)
    df["incident_dow"]   = df["incident_date"].dt.dayofweek.astype(np.float32)

    return df


# ──────────────────────────────────────────────
# 2. Encode Features
# ──────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Encoding features ...")

    encoders = {}
    for col in ["incident_type", "incident_severity", "authorities_contacted",
                "insured_sex", "insured_education_level", "insured_occupation",
                "insured_hobbies", "collision_type", "incident_state",
                "policy_state", "police_report_available"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str)).astype(np.float32)
        encoders[col] = le

    # Log-transform monetary amounts to reduce skew
    df["claim_amount_log"]    = np.log1p(df["claim_amount"]).astype(np.float32)
    df["total_claim_log"]     = np.log1p(df["total_claim_amount"]).astype(np.float32)
    df["deductible_log"]      = np.log1p(df["policy_deductible"]).astype(np.float32)
    df["annual_premium_log"]  = np.log1p(df["policy_annual_premium"]).astype(np.float32)

    # Normalise hour to [0, 1]
    df["hour_norm"] = (df["incident_hour_of_the_day"] / 23.0).astype(np.float32)

    df["num_vehicles"]    = df["number_of_vehicles_involved"].astype(np.float32)
    df["bodily_injuries"] = df["bodily_injuries"].astype(np.float32)
    df["witnesses"]       = df["witnesses"].astype(np.float32)
    df["insured_age"]     = df["insured_age"].astype(np.float32)

    return df


# ──────────────────────────────────────────────
# 3. Build Node Mapping (policy_id + city bipartite)
# ──────────────────────────────────────────────

def build_node_mapping(df: pd.DataFrame):
    """
    Assign integer indices to:
      - policy_id  (prefix 'P:')
      - incident_city (prefix 'C:')
    Returns a dict: entity_str -> int index
    """
    print("[3/5] Building bipartite node mapping ...")

    policies = ["P:" + p for p in df["policy_id"].unique()]
    cities   = ["C:" + c for c in df["incident_city"].unique()]
    all_nodes = policies + cities

    node_map = {n: i for i, n in enumerate(all_nodes)}
    n_policy = len(policies)
    n_city   = len(cities)

    print(f"      Policy nodes : {n_policy:,}")
    print(f"      City nodes   : {n_city:,}")
    print(f"      Total nodes  : {len(node_map):,}")
    return node_map, n_policy, n_city


# ──────────────────────────────────────────────
# 4. Build PyG Data Object
# ──────────────────────────────────────────────

EDGE_FEATURE_COLS = [
    "claim_amount_log", "total_claim_log",
    "incident_type_enc", "incident_severity_enc",
    "hour_norm", "num_vehicles", "bodily_injuries", "witnesses",
    "police_report_available_enc", "authorities_contacted_enc",
    "deductible_log", "annual_premium_log",
    "insured_age", "incident_month", "incident_dow",
    "collision_type_enc", "insured_sex_enc",
    "insured_education_level_enc", "insured_occupation_enc",
]

def build_graph(df: pd.DataFrame, node_map: dict, n_policy: int, n_city: int) -> Data:
    print("[4/5] Building PyG Data object ...")
    n_nodes = len(node_map)

    # ── edge_index ──────────────────────────────
    src = df["policy_id"].map(lambda x: node_map["P:" + x]).values.astype(np.int64)
    dst = df["incident_city"].map(lambda x: node_map["C:" + x]).values.astype(np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # ── edge_attr ───────────────────────────────
    edge_attr = torch.tensor(df[EDGE_FEATURE_COLS].values, dtype=torch.float32)

    # ── edge_label ──────────────────────────────
    edge_label = torch.tensor(df["label"].values, dtype=torch.float32)

    # ── node features ───────────────────────────
    # Per-node statistics derived from all claims associated with that node.
    # Policy nodes: avg claim, fraud rate, num claims, avg hour
    # City nodes  : avg claim, fraud rate, num claims, incident type diversity
    out_deg = np.bincount(src, minlength=n_nodes).astype(np.float32)
    in_deg  = np.bincount(dst, minlength=n_nodes).astype(np.float32)

    # Weighted averages over edges
    claims  = df["claim_amount_log"].values.astype(np.float32)
    labels  = df["label"].values.astype(np.float32)
    hours   = df["hour_norm"].values.astype(np.float32)
    premiums = df["annual_premium_log"].values.astype(np.float32)

    def safe_avg(weights, indices, n):
        total = np.bincount(indices, weights=weights, minlength=n).astype(np.float32)
        count = np.bincount(indices, minlength=n).astype(np.float32)
        return np.divide(total, count, out=np.zeros(n, np.float32), where=count > 0)

    avg_claim_src   = safe_avg(claims,   src, n_nodes)
    avg_claim_dst   = safe_avg(claims,   dst, n_nodes)
    fraud_rate_src  = safe_avg(labels,   src, n_nodes)
    fraud_rate_dst  = safe_avg(labels,   dst, n_nodes)
    avg_hour_src    = safe_avg(hours,    src, n_nodes)
    avg_prem_src    = safe_avg(premiums, src, n_nodes)

    # Type diversity per city node
    type_div = np.zeros(n_nodes, np.float32)
    city_type_div = df.groupby(df["incident_city"].map(lambda x: node_map["C:" + x]))["incident_type_enc"].nunique()
    type_div[city_type_div.index.values] = city_type_div.values.astype(np.float32)

    x = torch.tensor(np.stack([
        np.log1p(out_deg),
        np.log1p(in_deg),
        avg_claim_src,
        avg_claim_dst,
        fraud_rate_src,
        fraud_rate_dst,
        avg_hour_src,
        avg_prem_src,
        type_div,
    ], axis=1), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        num_nodes=n_nodes,
    )

    fraud_n = int(edge_label.sum().item())
    print(f"      Nodes        : {data.num_nodes:,}")
    print(f"      Edges        : {data.num_edges:,}")
    print(f"      Node features: {data.x.shape[1]}")
    print(f"      Edge features: {data.edge_attr.shape[1]}")
    print(f"      Fraud edges  : {fraud_n:,} ({fraud_n/data.num_edges:.4%})")
    return data


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    csv_path = os.path.join(DATA_DIR, CSV_NAME)
    if not os.path.isfile(csv_path):
        print(f"\n[ERROR] Dataset not found: {csv_path}")
        sys.exit(1)

    df                         = load_and_clean(csv_path)
    df                         = encode_features(df)
    node_map, n_policy, n_city = build_node_mapping(df)
    data                       = build_graph(df, node_map, n_policy, n_city)

    out_path = os.path.join(DATA_DIR, OUT_NAME)
    print(f"[5/5] Saving graph → {out_path}")
    torch.save(data, out_path)
    print("      Done ✓")


if __name__ == "__main__":
    main()
