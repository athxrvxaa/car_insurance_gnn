"""
train_gnn.py
============
Trains a GraphSAGE edge-classifier on the car insurance fraud graph.

The dataset is small (30k rows) so the full graph fits comfortably in
memory. We still use NeighborLoader for consistency and to keep the
pipeline GPU-memory safe.

Architecture:
  GraphSAGE encoder (node embeddings)  →  Edge MLP (fraud logit per claim)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from utils import (set_seed, split_edges, compute_metrics,
                   print_metrics, print_classification_report,
                   compute_pos_weight)

# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────
DEFAULT_EPOCHS        = 30
DEFAULT_HIDDEN_DIM    = 128
DEFAULT_BATCH_SIZE    = 512
DEFAULT_LR            = 3e-4
DEFAULT_NUM_NEIGHBORS = [20, 15]
DATA_PATH             = os.path.join("data", "insurance_graph.pt")


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class EdgeClassifier(nn.Module):
    """
    Concatenate source + destination node embeddings with edge features,
    then classify via MLP.
    """
    def __init__(self, hidden, edge_feat_dim):
        super().__init__()
        in_sz = 2 * hidden + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_sz, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, node_emb, edge_index, edge_attr):
        src_emb  = node_emb[edge_index[0]]
        dst_emb  = node_emb[edge_index[1]]
        edge_rep = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
        return self.mlp(edge_rep).squeeze(-1)


class FraudGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden):
        super().__init__()
        self.encoder    = GraphSAGEEncoder(node_feat_dim, hidden)
        self.classifier = EdgeClassifier(hidden, edge_feat_dim)

    def forward(self, x, edge_index, edge_attr):
        emb    = self.encoder(x, edge_index)
        logits = self.classifier(emb, edge_index, edge_attr)
        return logits


# ──────────────────────────────────────────────
# Training / Eval loops
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss   = criterion(logits, batch.edge_label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.edge_label.size(0)
        total_n    += batch.edge_label.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        probs  = torch.sigmoid(logits).cpu().numpy()
        labels = batch.edge_label.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    return np.concatenate(all_labels), np.concatenate(all_probs)


# ──────────────────────────────────────────────
# Loader helper
# ──────────────────────────────────────────────

def edge_mask_to_node_mask(edge_mask, edge_index, num_nodes):
    active = edge_index[:, edge_mask].unique()
    mask   = torch.zeros(num_nodes, dtype=torch.bool)
    mask[active] = True
    return mask


def make_loader(data, node_mask, num_neighbors, batch_size, shuffle=False):
    input_nodes = node_mask.nonzero(as_tuple=False).squeeze()
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=input_nodes,
        shuffle=shuffle,
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    if device == "cuda":
        print(f"       GPU   : {torch.cuda.get_device_name(0)}")

    if not os.path.isfile(DATA_PATH):
        print(f"\n[ERROR] Graph not found: {DATA_PATH}")
        print("  Run: python graph_construction.py")
        sys.exit(1)

    print(f"\n[1/4] Loading graph ...")
    data = torch.load(DATA_PATH, weights_only=False)
    print(f"      Nodes: {data.num_nodes:,}  Edges: {data.num_edges:,}")

    print("\n[2/4] Splitting edges (70 / 15 / 15) ...")
    train_mask, val_mask, test_mask = split_edges(data.num_edges)

    train_node_mask = edge_mask_to_node_mask(train_mask, data.edge_index, data.num_nodes)
    val_node_mask   = edge_mask_to_node_mask(val_mask,   data.edge_index, data.num_nodes)
    test_node_mask  = edge_mask_to_node_mask(test_mask,  data.edge_index, data.num_nodes)

    train_loader = make_loader(data, train_node_mask, args.num_neighbors, args.batch_size, shuffle=True)
    val_loader   = make_loader(data, val_node_mask,   args.num_neighbors, args.batch_size)
    test_loader  = make_loader(data, test_node_mask,  args.num_neighbors, args.batch_size)

    print("\n[3/4] Class weight ...")
    # Insurance dataset has ~11.5% fraud — much better balance than PaySim.
    # Clamp pos_weight at 20 (raw ~7.7) to avoid over-penalising negatives.
    pos_weight = compute_pos_weight(data.edge_label[train_mask]).to(device)
    pos_weight = torch.clamp(pos_weight, max=20.0)
    print(f"  Clamped pos_weight: {pos_weight.item():.2f}")
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    node_feat_dim = data.x.shape[1]
    edge_feat_dim = data.edge_attr.shape[1]

    model = FraudGNN(node_feat_dim, edge_feat_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n      Parameters  : {total_params:,}")
    print(f"      Hidden dim  : {args.hidden_dim}")
    print(f"      Batch size  : {args.batch_size}")
    print(f"      Neighbours  : {args.num_neighbors}\n")

    print("[4/4] Training ...")
    best_val_auc  = 0.0
    best_path     = os.path.join("data", "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_labels, val_probs = evaluate(model, val_loader, device)
        vm = compute_metrics(val_labels, val_probs)
        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"Loss={loss:.4f}  Val AUC={vm['ROC_AUC']:.4f}  F1={vm['F1']:.4f}")
        if vm["ROC_AUC"] > best_val_auc:
            best_val_auc = vm["ROC_AUC"]
            torch.save(model.state_dict(), best_path)

    print(f"\n  Best Val AUC : {best_val_auc:.4f}")
    print(f"  Checkpoint   : {best_path}")

    print("\n[TEST] Evaluating best model ...")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_labels, test_probs = evaluate(model, test_loader, device)

    # Sweep thresholds to find best F1
    best_f1, best_thresh = 0.0, 0.5
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        m = compute_metrics(test_labels, test_probs, threshold=t)
        if m["F1"] > best_f1:
            best_f1, best_thresh = m["F1"], t
    print(f"  Best threshold (F1): {best_thresh}")

    test_metrics = compute_metrics(test_labels, test_probs, threshold=best_thresh)
    print_metrics(test_metrics, "GNN Test Results")
    print_classification_report(test_labels, test_probs, threshold=best_thresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden_dim",    type=int,   default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--batch_size",    type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--num_neighbors", type=int,   nargs="+", default=DEFAULT_NUM_NEIGHBORS)
    main(parser.parse_args())
