"""
train_diversification_predictor.py  –  Vynora 2.0
Train the Paper 5 Graph Neural Network Diversification Score Predictor.
Reference: Paper 5 Section 4.5 (Chun et al., Expert Systems with Applications 2024)

Architecture:
  - Node Feature Embedding: h_0 = W_embed * r_i (Eq 3)
  - Feature-Augmented Attention Propagation (Eq 4-6)
  - Prediction MLP: S_hat = W_2(sigma(W_1 h_i)) (Eq 7)
  - Optimization with MAE Loss (Eq 8)

Usage:
    python train_diversification_predictor.py --dataset takatak --epochs 30
    python train_diversification_predictor.py --dataset wechat --epochs 30
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
import config
from dataloader import VynoraDataset
from predict import load_model_and_dataset
from diversity import (
    DiversificationGNN,
    compute_empirical_diversification_scores
)


def build_video_cooccurrence_graph(dataset: VynoraDataset, max_neighbors: int = 15):
    """
    Build video-video co-occurrence network from user watch sessions
    as specified in Paper 5 Section 4.2 & 4.3.

    For each edge (i, j):
      edge_attr = [frequency, norm_rank, distance, closeness] (4-dim, Paper 5 Sec 4.5.2)
    """
    print("Building video co-occurrence graph for Paper 5 GNN...")
    allPos_video, _ = dataset.allPos
    vloggers = dataset.vlogger_list
    n_videos = dataset.n_videos

    # Track co-occurrence counts: (src, dst) -> count
    from collections import defaultdict
    pair_counts = defaultdict(int)

    for u, vids in enumerate(allPos_video):
        # Sample or use first 30 videos per user for efficiency
        session = vids[:30] if len(vids) > 30 else vids
        n = len(session)
        for i in range(n - 1):
            src, dst = session[i], session[i + 1]
            if src != dst:
                pair_counts[(src, dst)] += 1

    # Keep top neighbors per video
    adj_dict = defaultdict(list)
    for (src, dst), count in pair_counts.items():
        adj_dict[src].append((dst, count))

    edge_src, edge_dst, edge_feats = [], [], []
    for src in range(n_videos):
        neighbors = sorted(adj_dict[src], key=lambda x: -x[1])[:max_neighbors]
        max_c = neighbors[0][1] if neighbors else 1
        for rank, (dst, count) in enumerate(neighbors, start=1):
            edge_src.append(src)
            edge_dst.append(dst)
            freq = count / max_c
            rank_feat = 1.0 / rank
            dist = 1.0 if vloggers[src] != vloggers[dst] else 0.0
            closeness = 1.0 - dist
            edge_feats.append([freq, rank_feat, dist, closeness])

    edge_index = torch.LongTensor([edge_src, edge_dst]).to(config.DEVICE)
    edge_attr = torch.FloatTensor(edge_feats).to(config.DEVICE)
    print(f"Graph constructed: {n_videos} nodes, {edge_index.size(1)} directed edges.\n")
    return edge_index, edge_attr


def train_predictor(dataset_name: str = config.DATASET,
                    diversity_type: str = "publisher",
                    epochs: int = 30,
                    lr: float = 1e-3,
                    hidden_dim: int = 64):
    print(f"\n{'='*70}")
    print(f"  Training Paper 5 Diversification Score Predictor ({diversity_type})")
    print(f"  Dataset: {dataset_name} | Epochs: {epochs} | LR: {lr}")
    print(f"{'='*70}\n")

    # Load base model & dataset to obtain node representations
    model, dataset, graph_uv, graph_ua = load_model_and_dataset(dataset_name)

    # 1. Obtain base representations r_i
    with torch.no_grad():
        emb = model.forward(graph_uv, graph_ua)
        node_feats = emb["video_v"].detach() # [n_videos, D*(L+1)]

    # 2. Compute true empirical diversification scores (Paper 5 Section 3)
    target_scores = compute_empirical_diversification_scores(
        dataset, diversity_type=diversity_type, video_embeddings=node_feats
    )
    target_tensor = torch.FloatTensor(target_scores).to(config.DEVICE)

    # 3. Build graph
    edge_index, edge_attr = build_video_cooccurrence_graph(dataset)

    # 4. Instantiate Paper 5 DiversificationGNN
    predictor = DiversificationGNN(
        in_features_dim=node_feats.size(1),
        hidden_dim=hidden_dim,
        edge_dim=4,
        n_layers=2
    ).to(config.DEVICE)

    optimizer = optim.Adam(predictor.parameters(), lr=lr, weight_decay=1e-5)

    # Train / Val split (80% / 20%)
    n_nodes = dataset.n_videos
    indices = np.random.RandomState(42).permutation(n_nodes)
    train_idx = torch.LongTensor(indices[:int(0.8 * n_nodes)]).to(config.DEVICE)
    val_idx   = torch.LongTensor(indices[int(0.8 * n_nodes):]).to(config.DEVICE)

    best_val_mae = float("inf")
    ckpt_path = os.path.join(config.CKPT_DIR, f"diversification_gnn_{dataset_name}_{diversity_type}.pt")

    for epoch in range(1, epochs + 1):
        predictor.train()
        optimizer.zero_grad()

        pred_scores = predictor(node_feats, edge_index, edge_attr)
        train_loss = predictor.compute_loss(pred_scores[train_idx], target_tensor[train_idx])
        train_loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = predictor.compute_loss(pred_scores[val_idx], target_tensor[val_idx])

        if val_loss.item() < best_val_mae:
            best_val_mae = val_loss.item()
            torch.save({
                "epoch": epoch,
                "model_state": predictor.state_dict(),
                "val_mae": best_val_mae,
                "diversity_type": diversity_type
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:>2}/{epochs} | Train MAE: {train_loss.item():.4f} | Val MAE: {val_loss.item():.4f} (Best: {best_val_mae:.4f})")

    print(f"\n[Done] Trained Diversification Predictor saved to: {ckpt_path}\n")
    return predictor


def main():
    parser = argparse.ArgumentParser(description="Train Paper 5 Diversification Score Predictor")
    parser.add_argument("--dataset", type=str, default=config.DATASET, help="takatak | wechat")
    parser.add_argument("--type", type=str, default="publisher", choices=["publisher", "content"])
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_predictor(
        dataset_name=args.dataset,
        diversity_type=args.type,
        epochs=args.epochs,
        lr=args.lr
    )


if __name__ == "__main__":
    main()

