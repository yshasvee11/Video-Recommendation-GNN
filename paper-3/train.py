"""
train.py – Training Pipeline for MovieLens-1M Paper 3 Benchmark.
Supports training on GPU/CPU, CONDE graph denoising, dual checkpointing,
and generates the final publication comparison table.
"""

import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
import config
from dataloader import MovieLensDataset
from denoiser import BipartiteGraphDenoiser
from model import VynoraPaper3GNN
from evaluate import evaluate_model


def sample_bpr_batch(train_u, train_i, n_items, train_pos_set, batch_size=config.BATCH_SIZE):
    """
    Uniform negative sampling for BPR loss.
    """
    indices = np.random.choice(len(train_u), batch_size, replace=False)
    batch_u = train_u[indices]
    batch_pos = train_i[indices]
    
    batch_neg = []
    for u in batch_u:
        pos_set = train_pos_set[u]
        while True:
            neg = np.random.randint(0, n_items)
            if neg not in pos_set:
                batch_neg.append(neg)
                break
    batch_neg = np.array(batch_neg)
    
    return (
        torch.from_numpy(batch_u).long().to(config.DEVICE),
        torch.from_numpy(batch_pos).long().to(config.DEVICE),
        torch.from_numpy(batch_neg).long().to(config.DEVICE)
    )


def train(resume=True):
    print("=" * 70)
    print("VYNORA PAPER 3 BENCHMARK (MOVIELENS-1M)")
    print(f"Device: {config.DEVICE} | Dim: {config.EMBED_DIM} | Epochs: {config.EPOCHS}")
    print("=" * 70)
    
    # 1. Load Dataset
    dataset = MovieLensDataset()
    
    # 2. Initialize Model
    model = VynoraPaper3GNN(dataset.n_users, dataset.n_items).to(config.DEVICE)
    model.set_graphs(dataset.graph_ui, dataset.meta_graph_u, dataset.meta_graph_i)
    
    # 3. Initialize Denoiser
    denoiser = BipartiteGraphDenoiser(dataset.train_u, dataset.train_i, dataset.n_users, dataset.n_items)
    
    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    n_batches = len(dataset.train_u) // config.BATCH_SIZE
    best_val_ndcg = -1.0
    best_epoch = 0
    start_epoch = 1
    
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_movielens_model.pt")
    last_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "last_movielens_model.pt")
    latest_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_latest.pt")
    
    # Resume from checkpoint if available
    if resume and os.path.exists(latest_ckpt_path):
        print(f"\n[RESUME DETECTED] Loading checkpoint from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_ndcg = ckpt.get("best_val_ndcg", -1.0)
        best_epoch = ckpt.get("best_epoch", 0)
        if "denoiser_scores_ema" in ckpt:
            denoiser.edge_scores_ema = ckpt["denoiser_scores_ema"]
            denoiser.active_mask = ckpt["denoiser_active_mask"]
            # Rebuild graph according to resumed mask
            kept_u = denoiser.train_u[denoiser.active_mask]
            kept_i = denoiser.train_i[denoiser.active_mask]
            pruned_graph = dataset.build_bipartite_adj(kept_u, kept_i)
            model.set_graphs(pruned_graph, dataset.meta_graph_u, dataset.meta_graph_i)
        print(f"Resumed successfully! Next training epoch: {start_epoch}/{config.EPOCHS} | Best Val NDCG so far: {best_val_ndcg:.4f} (Epoch {best_epoch})")
    
    if start_epoch > config.EPOCHS:
        print(f"Model has already completed all {config.EPOCHS} epochs.")
    else:
        print(f"\nStarting Training from Epoch {start_epoch} to {config.EPOCHS}...")
        for epoch in range(start_epoch, config.EPOCHS + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            total_bpr = 0.0
            total_cl = 0.0
            
            # Periodic Graph Denoising
            if epoch > 1 and epoch % config.DENOISE_INTERVAL == 0:
                pruned_graph, n_retained, n_dropped = denoiser.update_and_prune(model, dataset)
                model.set_graphs(pruned_graph, dataset.meta_graph_u, dataset.meta_graph_i)
                print(f"  [Epoch {epoch} Denoising] Retained: {n_retained:,} | Dropped: {n_dropped:,} noisy edges")
                
            for _ in range(n_batches):
                users, pos_items, neg_items = sample_bpr_batch(
                    dataset.train_u, dataset.train_i, dataset.n_items, dataset.train_pos_set
                )
                
                optimizer.zero_grad()
                h_u, h_v = model.forward()
                
                bpr_loss, reg_loss = model.compute_bpr_loss(h_u, h_v, users, pos_items, neg_items)
                cl_loss = model.compute_contrastive_loss(users, pos_items)
                
                loss = (
                    config.GAMMA_SUPERVISED * bpr_loss +
                    config.BETA_CONTRASTIVE * cl_loss +
                    reg_loss
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_bpr += bpr_loss.item()
                total_cl += cl_loss.item()
                
            epoch_time = time.time() - t0
            
            # Save latest checkpoint after every epoch for seamless recovery
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_ndcg": best_val_ndcg,
                "best_epoch": best_epoch,
                "denoiser_scores_ema": denoiser.edge_scores_ema,
                "denoiser_active_mask": denoiser.active_mask
            }, latest_ckpt_path)
            
            # Evaluation every 5 epochs or final epoch
            if epoch % 5 == 0 or epoch == config.EPOCHS:
                val_metrics = evaluate_model(model, dataset, eval_mode="val", sample_users=1000)
                print(
                    f"Epoch {epoch:02d}/{config.EPOCHS:02d} ({epoch_time:.1f}s) | "
                    f"Loss: {total_loss / n_batches:.4f} | "
                    f"Val Prec: {val_metrics['Precision@10']:.4f} | "
                    f"Val Rec: {val_metrics['Recall@10']:.4f} | "
                    f"Val NDCG: {val_metrics['NDCG@10']:.4f} | "
                    f"Val AUC: {val_metrics['AUC']:.4f}"
                )
                
                if val_metrics["NDCG@10"] > best_val_ndcg:
                    best_val_ndcg = val_metrics["NDCG@10"]
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f"  -> Saved new best checkpoint to {best_ckpt_path}")
            else:
                print(f"Epoch {epoch:02d}/{config.EPOCHS:02d} ({epoch_time:.1f}s) | Loss: {total_loss / n_batches:.4f}")
                
        # Save last epoch checkpoint
        torch.save(model.state_dict(), last_ckpt_path)
        print(f"\nTraining completed. Saved last checkpoint to {last_ckpt_path}")
        print(f"Best validation epoch was {best_epoch} with NDCG@10 = {best_val_ndcg:.4f}")
    
    # Final Test Evaluation with Best Checkpoint
    model.load_state_dict(torch.load(best_ckpt_path, map_location=config.DEVICE, weights_only=False), strict=False)
    print("\nRunning Final Test Evaluation with lambda=0.0 (Pure Accuracy vs Paper 3)...")
    test_metrics_pure = evaluate_model(model, dataset, eval_mode="test", lambda_div=0.0)
    
    print("\nRunning Final Test Evaluation with lambda=0.3 (Genre Diversity Enabled)...")
    test_metrics_div = evaluate_model(model, dataset, eval_mode="test", lambda_div=0.3)
    
    # Save Results JSON
    results = {
        "dataset": "MovieLens-1M",
        "embed_dim": config.EMBED_DIM,
        "epochs": config.EPOCHS,
        "best_epoch": best_epoch,
        "test_pure_accuracy": test_metrics_pure,
        "test_with_diversity": test_metrics_div
    }
    results_path = os.path.join(config.RESULTS_DIR, "movielens_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to {results_path}")
    
    # Generate Comparison Table
    generate_comparison_markdown(test_metrics_pure, test_metrics_div)


def generate_comparison_markdown(pure_res, div_res):
    """
    Produces publication-ready comparison table against Paper 3 Table 5.
    """
    table_md = f"""# Final Comparison: Vynora vs. Paper 3 (MPGCL) on MovieLens-1M

**Benchmark Protocol**: MovieLens-1M | 80% Train : 10% Val : 10% Test | K = 10 | d = 200

| Model | Precision@10 | Recall@10 | NDCG@10 | AUC | Publication Reference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DeepFM** | 0.2323 | 0.3008 | 0.3991 | 0.6883 | Guo et al., IJCAI 2017 |
| **GraphSAGE** | 0.3172 | 0.3491 | 0.4100 | 0.7290 | Hamilton et al., NeurIPS 2017 |
| **MMGCN** | 0.3555 | 0.3889 | 0.4319 | 0.7383 | Wei et al., MM 2019 |
| **LightGCN** | 0.3693 | 0.4198 | 0.4474 | 0.7499 | He et al., SIGIR 2020 |
| **UltraGCN** | 0.3782 | 0.4239 | 0.4599 | 0.7681 | Mao et al., CIKM 2021 |
| **HAN** | 0.3774 | 0.4295 | 0.4691 | 0.7661 | Wang et al., WWW 2019 |
| **HetGNN** | 0.3812 | 0.4294 | 0.4784 | 0.7682 | Zhang et al., KDD 2019 |
| **GCL** | 0.3448 | 0.3994 | 0.4223 | 0.7094 | You et al., NeurIPS 2020 |
| **HHFAN** | 0.3803 | 0.4405 | 0.4813 | 0.7790 | ESWA Baseline |
| **MPGCL (Paper 3 SOTA)** | **0.4091** | **0.4636** | **0.5038** | **0.8121** | He et al., ESWA 2023 |
| **Vynora (lambda=0.0)** | **{pure_res['Precision@10']:.4f}** | **{pure_res['Recall@10']:.4f}** | **{pure_res['NDCG@10']:.4f}** | **{pure_res['AUC']:.4f}** | **Our Proposed Framework** |
| **Vynora (lambda=0.3)** | **{div_res['Precision@10']:.4f}** | **{div_res['Recall@10']:.4f}** | **{div_res['NDCG@10']:.4f}** | **{div_res['AUC']:.4f}** | **Our Proposed (Genre Diversified)** |
"""
    out_path = os.path.join(config.RESULTS_DIR, "movielens_final_comparison_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table_md)
    print(f"\nGenerated comparison markdown table at: {out_path}")
    print(table_md)


if __name__ == "__main__":
    train()
