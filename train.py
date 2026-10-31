"""
train.py  –  Vynora 2.0
Full training loop with:
  - Dual-stream BPR sampling (VA-GNN pattern)
  - Cross-view contrastive loss (MPGCL)
  - Denoising consistency loss
  - CONDE-style periodic graph denoising (every D epochs)
  - Early stopping on Recall@10 (validation)
  - Checkpointing best model
"""

import os
import sys
import random
import time
import json
import numpy as np
import torch
from torch.optim import Adam

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from dataloader import VynoraDataset
from model import VynoraGNN
from sampler import uniform_sample, shuffle, minibatch
from denoiser import GraphDenoiser
from evaluate import evaluate, format_results


# ───────────────────────────────────────────────────────────────
# Reproducibility
# ───────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ───────────────────────────────────────────────────────────────
# Main training function
# ───────────────────────────────────────────────────────────────
def train():
    set_seed(config.SEED)
    if config.DEVICE.type == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {config.DEVICE}")
    print(f"Dataset: {config.DATASET}\n")

    # ── Load dataset ─────────────────────────────────────────────
    dataset = VynoraDataset(config.DATASET)

    # ── Initialise model ─────────────────────────────────────────
    model = VynoraGNN(
        n_users   = dataset.n_users,
        n_videos  = dataset.n_videos,
        n_vloggers= dataset.n_vloggers,
        vlogger_list = dataset.vlogger_list
    ).to(config.DEVICE)

    optimizer = Adam(model.parameters(), lr=config.LR)

    # ── Build initial sparse graphs ───────────────────────────────
    graph_uv = dataset.get_sparse_graph_uv()
    graph_ua = dataset.get_sparse_graph_ua()

    # ── CONDE denoisers ───────────────────────────────────────────
    # Hyperparameters adapted from CONDE's best configs (ciao/yelp)
    DENOISE_BETA    = 0.5
    DENOISE_D       = 10     # denoise every 10 epochs
    DENOISE_R       = 0.5
    DENOISE_GAMMA   = 1.0
    DENOISE_EPSILON = 5      # min neighbourhood size

    denoiser_uv = GraphDenoiser(
        train_users = dataset._train_uv_users,
        train_items = dataset._train_uv_videos,
        n_users     = dataset.n_users,
        n_items     = dataset.n_videos,
        beta=DENOISE_BETA, D=DENOISE_D, R=DENOISE_R,
        gamma=DENOISE_GAMMA, epsilon=DENOISE_EPSILON
    )
    denoiser_ua = GraphDenoiser(
        train_users = dataset._train_ua_users,
        train_items = dataset._train_ua_vloggers,
        n_users     = dataset.n_users,
        n_items     = dataset.n_vloggers,
        beta=DENOISE_BETA, D=DENOISE_D, R=DENOISE_R,
        gamma=DENOISE_GAMMA, epsilon=DENOISE_EPSILON
    )

    # ── Training loop ─────────────────────────────────────────────
    best_recall  = -1.0
    best_epoch   = 0
    patience_cnt = 0
    history      = []

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        model.train()

        # Fetch current (possibly denoised) graphs
        graph_uv = denoiser_uv.graph
        graph_ua = denoiser_ua.graph

        # BPR sampling with Sagittarius frequency weights
        S, S2, W = uniform_sample(dataset, config.NEG_K)
        if len(S) == 0:
            print(f"[Epoch {epoch}] WARNING: empty training batch - skipping")
            continue

        # Convert to tensors
        users       = torch.LongTensor(S[:, 0]).to(config.DEVICE)
        pos_videos  = torch.LongTensor(S[:, 1]).to(config.DEVICE)
        neg_videos  = torch.LongTensor(S[:, 2]).to(config.DEVICE)
        pos_vloggers= torch.LongTensor(S[:, 3]).to(config.DEVICE)
        neg_vloggers= torch.LongTensor(S[:, 4]).to(config.DEVICE)
        weights     = torch.FloatTensor(W).to(config.DEVICE)

        (users, pos_videos, neg_videos,
         pos_vloggers, neg_vloggers, weights) = shuffle(users, pos_videos, neg_videos,
                                                       pos_vloggers, neg_vloggers, weights)

        users2        = torch.LongTensor(S2[:, 0]).to(config.DEVICE) if len(S2) > 0 else None
        pos_vloggers2 = torch.LongTensor(S2[:, 1]).to(config.DEVICE) if len(S2) > 0 else None
        neg_vloggers2 = torch.LongTensor(S2[:, 2]).to(config.DEVICE) if len(S2) > 0 else None

        # Mini-batch training
        total_loss = 0.0
        n_batches  = 0

        batches = list(minibatch(users, pos_videos, neg_videos,
                                 pos_vloggers, neg_vloggers, weights,
                                 batch_size=config.BPR_BATCH))

        for batch_idx, (u_b, pv_b, nv_b, pa_b, na_b, w_b) in enumerate(batches):
            optimizer.zero_grad()

            # Forward pass
            emb = model.forward(graph_uv, graph_ua)
            user_v    = emb["user_v"]
            video_v   = emb["video_v"]
            user_a    = emb["user_a"]
            vlogger_a = emb["vlogger_a"]

            # L_rank: primary BPR with frequency weights (Sagittarius adaptation)
            loss_bpr = model.bpr_loss_uv(
                u_b, pv_b, nv_b, pa_b, na_b,
                user_v, video_v, user_a, vlogger_a,
                weights=w_b
            )

            # L_cl: VA-GNN 3-Entity cross-view contrastive loss (Users, Videos, Creators)
            loss_cl = model.contrastive_loss(
                user_v, user_a, u_b,
                video_v=video_v, vlogger_a=vlogger_a,
                batch_videos=pv_b, batch_vloggers=pa_b
            )

            # L_den: denoising consistency
            loss_den = model.denoising_consistency_loss(user_v, user_a, u_b)

            loss = loss_bpr + config.CL_REG * loss_cl + config.DENOISE_REG * loss_den

            # L_vlogger: auxiliary user-vlogger BPR (delta * L_vlogger)
            if users2 is not None and len(users2) > 0:
                # Use same batch indices (modulo size)
                idx = torch.randint(0, len(users2), (len(u_b),))
                loss_ua = model.bpr_loss_ua(
                    users2[idx], pos_vloggers2[idx], neg_vloggers2[idx],
                    user_a, vlogger_a
                )
                loss = loss + config.VLOGGER_REG * loss_ua

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(batches):
                print(f"\r  [Epoch {epoch:2d}] Batch {batch_idx+1:3d}/{len(batches)} | Loss: {loss.item():.4f}", end="", flush=True)

        avg_loss = total_loss / max(n_batches, 1)
        print()

        # ── CONDE denoising step (every D epochs) ─────────────────
        if epoch % DENOISE_D == 0:
            model.eval()
            with torch.no_grad():
                # Score UV edges
                eu = torch.LongTensor(dataset._train_uv_users).to(config.DEVICE)
                ev = torch.LongTensor(dataset._train_uv_videos).to(config.DEVICE)
                scores_uv = model.compute_edge_reliability_uv(eu, ev).cpu().numpy()
                denoiser_uv.update(scores_uv)

                # Score UA edges
                eu2 = torch.LongTensor(dataset._train_ua_users).to(config.DEVICE)
                ea  = torch.LongTensor(dataset._train_ua_vloggers).to(config.DEVICE)
                scores_ua = model.compute_edge_reliability_ua(eu2, ea).cpu().numpy()
                denoiser_ua.update(scores_ua)

            # Apply drop rule
            ret_uv, drop_uv = denoiser_uv.denoise()
            ret_ua, drop_ua = denoiser_ua.denoise()
            print(f"  [Denoise] UV: retained={ret_uv} dropped={drop_uv} | "
                  f"UA: retained={ret_ua} dropped={drop_ua}")
            model.train()

        # ── Validation ────────────────────────────────────────────
        val_results = evaluate(model, dataset, denoiser_uv.graph, denoiser_ua.graph,
                               mode="val")
        recall_at_k = val_results.get("Recall", {})
        recall10    = recall_at_k.get(config.TOPK_LIST[0], 0.0)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{config.EPOCHS} | loss={avg_loss:.4f} | "
              f"Val Recall@{config.TOPK_LIST[0]}={recall10:.4f} | {elapsed:.1f}s")

        row = {"epoch": epoch, "loss": avg_loss}
        for m, vals in val_results.items():
            for k, v in vals.items():
                row[f"val_{m}@{k}"] = v
        history.append(row)

        # ── Early stopping & checkpointing ────────────────────────
        if recall10 > best_recall:
            best_recall  = recall10
            best_epoch   = epoch
            patience_cnt = 0
            ckpt_path = os.path.join(config.CKPT_DIR, f"best_{config.DATASET}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_recall":  recall10,
                "val_results": val_results,
            }, ckpt_path)
            print(f"  [+] New best Recall@{config.TOPK_LIST[0]}={recall10:.4f} -> saved to {ckpt_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= config.PATIENCE:
                print(f"\nEarly stop at epoch {epoch} (no improvement for {config.PATIENCE} epochs).")
                break

    # ── Final test evaluation ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Best val epoch: {best_epoch}  Recall@{config.TOPK_LIST[0]}={best_recall:.4f}")
    print("Loading best checkpoint for test evaluation...")

    ckpt = torch.load(os.path.join(config.CKPT_DIR, f"best_{config.DATASET}.pt"),
                      map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)

    test_results = evaluate(model, dataset, denoiser_uv.graph, denoiser_ua.graph,
                            mode="test")
    print("\nTest Results:")
    print(format_results(test_results))

    # Save results
    out = {
        "dataset":      config.DATASET,
        "best_epoch":   best_epoch,
        "val_recall":   best_recall,
        "test_results": {m: {str(k): v for k, v in vals.items()}
                         for m, vals in test_results.items()},
        "history":      history,
    }
    result_path = os.path.join(config.RESULT_DIR, f"results_{config.DATASET}.json")
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {result_path}")
    return test_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Vynora 2.0")
    parser.add_argument("--dataset", type=str, default=None, choices=["takatak", "wechat"],
                        help="Choose dataset to train: takatak | wechat (default from config.py)")
    args = parser.parse_args()
    if args.dataset:
        config.DATASET = args.dataset
    train()
