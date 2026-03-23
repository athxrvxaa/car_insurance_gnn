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

The dataset contains **30,000 claims** with an **11.47% fraud rate** — a much more realistic imbalance ratio than synthetic financial datasets.

---

## Graph Design

```
Nodes:
  Policy nodes   →  one per insured person (30,000)
  City nodes     →  one per incident city  (~17,931 unique)
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
│    → SAGEConv(9 → 128) + ReLU + Dropout(0.3)    │
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

**Training:**
- Loss: `BCEWithLogitsLoss` with `pos_weight` (clamped at 20)
- Optimiser: Adam lr=3e-4, weight_decay=1e-5
- Scheduler: CosineAnnealingLR
- Mini-batch: `NeighborLoader` — `[20, 15]` neighbours per hop
- Epochs: 30

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

### 2. Install PyTorch (CUDA 12.1 recommended)

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
python graph_construction.py
python baseline_models.py
python train_gnn.py --epochs 30 --batch_size 512 --hidden_dim 128 --lr 0.0003 --num_neighbors 20 15
or
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
  --lr             FLOAT   Learning rate              (default: 0.0003)
  --num_neighbors  INT...  Neighbours per hop         (default: 20 15)
```

---

## Dataset

**Car Insurance Fraud Dataset**

| Column | Description |
|---|---|
| `policy_id` | Unique policy identifier |
| `policy_state` | State where policy was issued |
| `policy_deductible` | Policy deductible amount |
| `policy_annual_premium` | Annual premium paid |
| `insured_age` | Age of insured person |
| `insured_sex` | Sex of insured |
| `insured_education_level` | Education level |
| `insured_occupation` | Occupation |
| `insured_hobbies` | Declared hobbies |
| `incident_date` | Date of incident |
| `incident_type` | Type of incident (Parked Car, Theft, etc.) |
| `collision_type` | Collision direction |
| `incident_severity` | Damage severity |
| `authorities_contacted` | Who was contacted (Police/Fire/Ambulance) |
| `incident_state` | State where incident occurred |
| `incident_city` | City where incident occurred |
| `incident_hour_of_the_day` | Hour of incident |
| `number_of_vehicles_involved` | Vehicles involved |
| `bodily_injuries` | Number of bodily injuries |
| `witnesses` | Number of witnesses |
| `police_report_available` | Whether police report exists |
| `claim_amount` | Amount claimed |
| `total_claim_amount` | Total claim including extras |
| `fraud_reported` | **Target** – Y/N |

---

## Why Bipartite Graph?

In insurance fraud, fraudsters often:
- Target the same cities repeatedly
- Use the same networks of repair shops / witnesses
- File claims with similar geographic patterns

By linking policy nodes to city nodes, GraphSAGE can propagate fraud signals through the graph — a policy connected to a high-fraud city inherits that signal in its embedding, even if its own claim appears legitimate in isolation.

---

## Key Differences from PaySim

| Aspect | PaySim | Car Insurance |
|---|---|---|
| Rows | 6.3M (sampled 1M) | 30,000 (full) |
| Fraud rate | 0.13% | 11.47% |
| Graph type | Account → Account | Policy → City (bipartite) |
| Node count | ~1.6M | ~48K |
| pos_weight | ~786 (clamped 100) | ~7.7 (clamped 20) |

---

## Future Improvements

- **Heterogeneous GNN** — add occupation and incident-type as additional node types
- **Link prediction** — detect suspiciously similar claim pairs (same city, same hour, same amount)
- **Temporal features** — model claim sequences per policy over time
- **SHAP explanations** — explain individual fraud predictions
- **Ensemble** — stack GNN embeddings with XGBoost predictions

---

## References

- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS.
- Fey, M. & Lenssen, J.E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
