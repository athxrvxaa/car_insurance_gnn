#  Car Insurance Fraud Detection using Graph Neural Networks

> MSc Data Science – Major Project | PyTorch Geometric · GraphSAGE · XGBoost

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.7-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project detects fraudulent car insurance claims using a **Graph Neural Network (GraphSAGE)** and compares it against an **XGBoost baseline**.

Insurance claims are modelled as a **bipartite directed graph**:
- **Policy nodes** → **Incident city nodes**, connected by claim edges
- GraphSAGE aggregates neighbourhood signals to detect fraud patterns that cluster by geography and policy behaviour

The dataset contains **30,000 claims** with an **11.47% fraud rate**.

### Results

| Model | ROC AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost (Baseline) | ~0.93 | — | — | — |
| **GraphSAGE (GNN)** | **0.9700** | **0.7770** | **0.8149** | **0.7955** |

> **The GNN significantly outperforms the tabular baseline** on this dataset. With `num_neighbors=155` (wide neighbourhood sampling), GraphSAGE achieves ROC AUC of **0.97** and F1 of **0.80** in just a single epoch — demonstrating that graph structure provides strong fraud signals that tabular models cannot capture.

---

## Graph Design

```
Nodes:
  Policy nodes   →  one per insured person  (30,000)
  City nodes     →  one per incident city   (~17,931 unique)
  Total nodes    :  ~47,931

Edges:
  One directed edge per claim: policy → incident_city
  Total edges: 30,000

Edge features (19):
  claim_amount_log, total_claim_log,
  incident_type, incident_severity, hour_of_day,
  num_vehicles, bodily_injuries, witnesses,
  police_report, authorities_contacted,
  deductible_log, annual_premium_log,
  insured_age, incident_month, day_of_week,
  collision_type, insured_sex,
  education_level, occupation

Node features (9):
  log(out_degree), log(in_degree),
  avg_claim_sent, avg_claim_received,
  fraud_rate_as_sender, fraud_rate_as_receiver,
  avg_hour_of_incident, avg_annual_premium,
  incident_type_diversity

Edge label:
  fraud_reported  { 1 = fraud, 0 = legitimate }
```

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│              GraphSAGE Encoder                   │
│  node_feats [N, 9]                               │
│    → SAGEConv(9 → 128) + ReLU + Dropout(0.3)     │
│    → SAGEConv(128 → 128)                         │
│  output: node embeddings [N, 128]                │
└──────────────────────────────────────────────────┘
                       │
                       ▼  per edge: [src_emb | dst_emb | edge_attr]
┌──────────────────────────────────────────────────┐
│              Edge MLP Classifier                 │
│  input [275] → Linear(275→128) + BN + ReLU + Drop│
│             → Linear(128→64) + ReLU              │
│             → Linear(64→1)                       │
│  output: fraud logit per claim edge              │
└──────────────────────────────────────────────────┘
```

**Best training configuration:**
- Loss: `BCEWithLogitsLoss` with `pos_weight=7.75` (raw ratio, no clamping needed)
- Optimiser: Adam lr=2e-4, weight_decay=1e-5
- Scheduler: CosineAnnealingLR
- Mini-batch: `NeighborLoader` — `[20, 155]` neighbours per hop
- Epochs: 1 (converges very fast with wide neighbourhood sampling)

> **Key insight:** Setting `num_neighbors` to a wide value (155 at hop-2) allows the GNN to aggregate across a much larger subgraph per batch, capturing the full fraud signal from city nodes that have many associated claims.

---

## Project Structure

```
car_insurance_gnn/
│
├── data/                              ← place CSV here (gitignored)
│   ├── car_insurance_fraud_dataset.csv
│   ├── insurance_graph.pt             ← generated
│   ├── best_model.pt                  ← GNN checkpoint
│   ├── xgb_model.json                 ← XGBoost model
│   └── xgb_feature_importance.png
│
├── graph_construction.py              ← Step 1: CSV → bipartite graph
├── train_gnn.py                       ← Step 2: GraphSAGE training
├── baseline_models.py                 ← Step 3: XGBoost baseline
├── utils.py                           ← shared metrics & helpers
├── requirements.txt
├── run_project.ps1                    ← Windows one-click runner
└── README.md
```

---

## Quickstart

### 1. Environment setup

```bash
git clone https://github.com/YOUR_USERNAME/car-insurance-gnn-fraud.git
cd car-insurance-gnn-fraud

conda create -n paysim_gnn python=3.10 -y
conda activate paysim_gnn
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PyTorch Geometric

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### 4. Install remaining packages

```bash
pip install pandas scikit-learn xgboost tqdm matplotlib "numpy<2"
```

### 5. Add the dataset

Place `car_insurance_fraud_dataset.csv` in the `data/` folder.

### 6. Run

```bash
# Step 1 – Build graph
python graph_construction.py

# Step 2 – XGBoost baseline
python baseline_models.py

# Step 3 – Train GNN (best config)
python train_gnn.py --epochs 1 --batch_size 512 --hidden_dim 128 --lr 0.0002 --num_neighbors 20 155
```

**Windows:**
```powershell
.\run_project.ps1
```

---

## CLI Reference

```
python train_gnn.py [options]

  --epochs         INT     Training epochs           (default: 30)
  --hidden_dim     INT     GNN hidden size            (default: 128)
  --batch_size     INT     Mini-batch size            (default: 512)
  --lr             FLOAT   Learning rate              (default: 0.0002)
  --num_neighbors  INT...  Neighbours per hop         (default: 20 155)
```

---

## Actual Output

<details>
<summary><b>graph_construction.py</b></summary>

```
[1/5] Loading dataset ...
      Raw shape: (30000, 24)
      Fraud rate: 11.4667%
[3/5] Building bipartite node mapping ...
      Policy nodes : 30,000
      City nodes   : 17,931
      Total nodes  : 47,931
[4/5] Building PyG Data object ...
      Nodes        : 47,931
      Edges        : 30,000
      Node features: 9
      Edge features: 19
      Fraud edges  : 3,440 (11.4667%)
[5/5] Saving graph → data\insurance_graph.pt  ✓
```
</details>

<details>
<summary><b>train_gnn.py (GraphSAGE) — best run</b></summary>

```
[INFO] Device: cuda
       GPU   : NVIDIA GeForce RTX 2050

[3/4] Class weight ...
  pos_weight = 7.75  (neg: 18,599, pos: 2,401)
  Clamped pos_weight: 7.75

      Parameters  : 79,233
      Hidden dim  : 128
      Batch size  : 512
      Neighbours  : [20, 155]

[4/4] Training ...
  Epoch   1/1  Loss=1.1756  Val AUC=0.9735  F1=0.8019

  Best Val AUC : 0.9735

[TEST] Evaluating best model ...
  Best threshold (F1): 0.5

==================================================
  GNN Test Results
==================================================
  ROC_AUC     : 0.9700
  Precision   : 0.7770
  Recall      : 0.8149
  F1          : 0.7955
==================================================

Classification Report:
              precision    recall  f1-score   support
       Legit       0.98      0.97      0.97      9127
       Fraud       0.78      0.81      0.80      1167
    accuracy                           0.95     10294

Confusion Matrix:
  TN=  8,854  FP=    273
  FN=    216  TP=    951
```
</details>

---

## Why `num_neighbors=155` Works So Well

The key insight is how city nodes accumulate fraud signal:

- Each **city node** is connected to many policies — some cities have 10–50+ claims
- With `num_neighbors=[20, 155]`, the 2-hop sampling reaches 155 neighbours of each 1-hop neighbour
- This means a policy node effectively "sees" most of the city's claim history in a single forward pass
- City nodes with high historical fraud rates pass that signal directly to the policy being evaluated

With narrow sampling (e.g. `[10, 10]`), most batches miss the city's fraud context entirely. With wide sampling, the GNN aggregates the full neighbourhood signal — which is where the fraud pattern lives in this bipartite graph.

---

## Comparison: Car Insurance vs PaySim (Companion Project)

| Aspect | PaySim (GNN) | Car Insurance (GNN) |
|---|---|---|
| Dataset size | 6.3M (sampled 1M) | 30,000 (full) |
| Fraud rate | 0.13% | 11.47% |
| Graph type | Account → Account | Policy → City (bipartite) |
| pos_weight | ~786 → clamped 100 | 7.75 (no clamping needed) |
| Best AUC | 0.838 | **0.970** |
| Best F1 | 0.069 | **0.796** |
| Epochs needed | 20 | **1** |

The car insurance dataset is far more tractable: healthier class balance, rich edge features, and meaningful geographic clustering of fraud through city nodes.

---

## Future Improvements

- **Heterogeneous GNN** — add occupation and incident-type as additional node types
- **More epochs** — the model converges fast; running 5–10 epochs may push AUC toward 0.99
- **Link prediction** — detect suspiciously similar claim pairs (same city, same hour, same amount)
- **Temporal features** — model claim sequences per policy over time
- **SHAP explanations** — explain individual fraud predictions for insurance adjusters
- **Ensemble** — stack GNN embeddings with XGBoost predictions for maximum performance

---

## References

- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS.
- Fey, M. & Lenssen, J.E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
