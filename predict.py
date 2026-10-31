"""
predict.py  –  Vynora 2.0
Home Screen Inference with Paper 5 Diversity Recommendation:
Given a user ID, return top-K recommended video IDs balanced for relevance and diversity.

Reference: Paper 5 (Chun et al., Expert Systems with Applications 2024)

Usage:
    python predict.py --user 42 --topk 10 --dataset takatak
    python predict.py --user 42 --topk 10 --diversity 0.4 --diversity_type publisher
    python predict.py --user 0 --topk 20 --dataset wechat --diversity 0.3 --diversity_method mmr
    python predict.py --user 5 --diversity 0.0      # pure relevance baseline (no diversity)
"""

import argparse
import json
import os
import torch
import numpy as np
import config
from dataloader import VynoraDataset
from model import VynoraGNN
from denoiser import GraphDenoiser
from diversity import (
    calculate_diversity_metrics,
    compute_empirical_diversification_scores,
    diverse_rerank
)


def load_model_and_dataset(dataset_name: str, ckpt_path=None, allow_untrained: bool = False):
    """Load dataset, rebuild graph, load checkpoint."""
    dataset = VynoraDataset(dataset_name)

    model = VynoraGNN(
        n_users   = dataset.n_users,
        n_videos  = dataset.n_videos,
        n_vloggers= dataset.n_vloggers,
        vlogger_list = dataset.vlogger_list
    ).to(config.DEVICE)

    if ckpt_path is None:
        ckpt_path = os.path.join(config.CKPT_DIR, f"best_{dataset_name}.pt")

    if not os.path.exists(ckpt_path):
        if allow_untrained:
            print(f"[Notice] No checkpoint found at {ckpt_path}. Using initialized model embeddings for demo preview.")
            model.eval()
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}.\n"
                f"Train the model first: python train.py (or pass --demo to preview with initialized weights)"
            )
    else:
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(val Recall@{config.TOPK_LIST[0]}={ckpt['val_recall']:.4f})")

    # Rebuild graphs (full adjacency for inference)
    graph_uv = dataset.get_sparse_graph_uv()
    graph_ua = dataset.get_sparse_graph_ua()

    return model, dataset, graph_uv, graph_ua


@torch.no_grad()
def recommend_for_user(user_id: int, model: VynoraGNN, dataset: VynoraDataset,
                       graph_uv, graph_ua, topk: int = 10,
                       exclude_seen: bool = True,
                       diversity_lambda: float = config.DIVERSITY_LAMBDA,
                       diversity_type: str = config.DIVERSITY_TYPE,
                       diversity_method: str = config.DIVERSITY_METHOD,
                       candidate_pool_size: int = config.CANDIDATE_POOL):
    """
    Return top-K video recommendations for a single user, with optional Paper 5 diversity.

    Args:
        user_id              : 0-indexed user ID
        model                : trained VynoraGNN
        dataset              : VynoraDataset
        graph_uv/ua          : current adjacency tensors
        topk                 : number of videos to return
        exclude_seen         : whether to mask training-positive videos
        diversity_lambda     : diversity weight in [0, 1] (0.0 = pure relevance)
        diversity_type       : "publisher" | "content" | "hybrid"
        diversity_method     : "mmr" | "score_rerank"
        candidate_pool_size  : size of candidate pool before re-ranking

    Returns:
        dict containing:
          - 'results': list of recommendation dicts
          - 'diversity_metrics': metrics for the returned list (Paper 5 Eq 1 & 2)
          - 'baseline_results': list of pure-relevance recommendations (for comparison)
          - 'baseline_diversity_metrics': metrics for baseline recommendations
    """
    if user_id < 0 or user_id >= dataset.n_users:
        raise ValueError(f"user_id {user_id} out of range [0, {dataset.n_users-1}]")

    emb = model.forward(graph_uv, graph_ua)
    user_v    = emb["user_v"]
    video_v   = emb["video_v"]
    user_a    = emb["user_a"]
    vlogger_a = emb["vlogger_a"]

    # Vlogger-video mapping
    vlogger_of_video = torch.LongTensor(dataset.vlogger_list).to(config.DEVICE)

    u_v = user_v[user_id].unsqueeze(0)   # [1, D*(L+1)]
    u_a = user_a[user_id].unsqueeze(0)

    s_vid  = torch.matmul(u_v, video_v.T).squeeze(0)       # [n_videos]
    p_of_v = vlogger_a[vlogger_of_video]                    # [n_videos, D*(L+1)]
    s_vlog = torch.matmul(u_a, p_of_v.T).squeeze(0)        # [n_videos]

    if getattr(config, "USE_ADAPTIVE_GATE", True):
        w_gate = model.compute_adaptive_gate(video_v, p_of_v)
        scores = (s_vid * w_gate + s_vlog * (1.0 - w_gate)).cpu().numpy()
    else:
        scores = (config.LAMBDA_VLOGGER * s_vid
                  + (1 - config.LAMBDA_VLOGGER) * s_vlog).cpu().numpy()

    # Mask seen items
    if exclude_seen:
        allPos_video, _ = dataset.allPos
        for v in allPos_video[user_id]:
            scores[v] = -1e9

    # 1. Base ranking (pure relevance)
    base_ranking = np.argsort(-scores)[:topk].tolist()
    base_results = []
    for rank, vid in enumerate(base_ranking, start=1):
        vlogger_id = int(dataset.vlogger_list[vid])
        base_results.append({
            "rank":       rank,
            "video_id":   vid,
            "vlogger_id": vlogger_id,
            "score":      float(scores[vid]),
        })

    video_v_np = video_v.detach().cpu().numpy()
    base_div_metrics = calculate_diversity_metrics(base_ranking, dataset, video_embeddings=video_v_np)

    # 2. Paper 5 Diversity Recommendation (if lambda > 0)
    if diversity_lambda > 0.0:
        pool_size = max(candidate_pool_size, topk * 2)
        candidate_ids = np.argsort(-scores)[:pool_size]
        candidate_scores = scores[candidate_ids]

        div_scores = None
        if diversity_method == "score_rerank":
            div_scores = compute_empirical_diversification_scores(
                dataset,
                diversity_type="content" if diversity_type == "content" else "publisher",
                video_embeddings=video_v_np
            )

        diversified_ids = diverse_rerank(
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            dataset=dataset,
            topk=topk,
            lambda_div=diversity_lambda,
            diversity_type=diversity_type,
            method=diversity_method,
            diversification_scores=div_scores,
            video_embeddings=video_v_np
        )

        results = []
        for rank, vid in enumerate(diversified_ids, start=1):
            vlogger_id = int(dataset.vlogger_list[vid])
            results.append({
                "rank":       rank,
                "video_id":   int(vid),
                "vlogger_id": vlogger_id,
                "score":      float(scores[vid]),
            })

        div_metrics = calculate_diversity_metrics(diversified_ids, dataset, video_embeddings=video_v_np)
    else:
        results = base_results
        div_metrics = base_div_metrics

    return {
        "results": results,
        "diversity_metrics": div_metrics,
        "baseline_results": base_results,
        "baseline_diversity_metrics": base_div_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Vynora 2.0 – Home Screen Recommender with Paper 5 Diversity")
    parser.add_argument("--user",             type=int,   default=42,                      help="User ID (0-indexed)")
    parser.add_argument("--topk",             type=int,   default=10,                     help="Number of recommendations")
    parser.add_argument("--dataset",          type=str,   default=config.DATASET,         help="takatak | wechat")
    parser.add_argument("--ckpt",             type=str,   default=None,                   help="Path to checkpoint file")
    parser.add_argument("--demo",             action="store_true",                        help="Allow running with initialized weights if no checkpoint is found")
    parser.add_argument("--no_exclude_seen",  action="store_true",                        help="Include already-seen videos in recommendations")
    parser.add_argument("--diversity", "--lambda_div", dest="diversity_lambda",
                        type=float, default=config.DIVERSITY_LAMBDA,                     help="Paper 5 diversity trade-off (0.0 = pure relevance, 1.0 = max diversity)")
    parser.add_argument("--diversity_type",   type=str,   default=config.DIVERSITY_TYPE,  choices=["publisher", "content", "hybrid"], help="Diversity dimension")
    parser.add_argument("--diversity_method", type=str,   default=config.DIVERSITY_METHOD, choices=["mmr", "score_rerank"], help="Paper 5 re-ranking method")
    args = parser.parse_args()

    model, dataset, graph_uv, graph_ua = load_model_and_dataset(args.dataset, args.ckpt, allow_untrained=args.demo)

    rec_data = recommend_for_user(
        user_id          = args.user,
        model            = model,
        dataset          = dataset,
        graph_uv         = graph_uv,
        graph_ua         = graph_ua,
        topk             = args.topk,
        exclude_seen     = not args.no_exclude_seen,
        diversity_lambda = args.diversity_lambda,
        diversity_type   = args.diversity_type,
        diversity_method = args.diversity_method
    )

    recs = rec_data["results"]
    div_m = rec_data["diversity_metrics"]
    base_div_m = rec_data["baseline_diversity_metrics"]

    div_label = f"Paper 5 Diversity (lambda={args.diversity_lambda}, {args.diversity_type}, {args.diversity_method})" if args.diversity_lambda > 0.0 else "Pure Relevance Baseline"
    print(f"\n{'='*56}")
    print(f"  Home Screen: Top-{args.topk} for User {args.user}  ({args.dataset})")
    print(f"  Mode: {div_label}")
    print(f"{'='*56}")
    print(f"{'Rank':>4}  {'Video ID':>8}  {'Vlogger':>8}  {'Score':>8}")
    print(f"{'-'*36}")
    for rec in recs:
        print(f"{rec['rank']:>4}  {rec['video_id']:>8}  "
              f"{rec['vlogger_id']:>8}  {rec['score']:>8.4f}")
    print(f"{'-'*36}")

    # Display Paper 5 Diversity Analytics
    print(f"\nPaper 5 Diversity Analysis (Top-{args.topk}):")
    print(f"  - Publisher Diversity (Eq 1):  {div_m['publisher_diversity']:.4f}")
    if div_m['content_diversity'] is not None:
        print(f"  - Content Diversity (Eq 1 & 2): {div_m['content_diversity']:.4f}")
    print(f"  - Unique Vloggers:             {div_m['unique_vloggers']} / {args.topk}")
    print(f"  - Vlogger Shannon Entropy:     {div_m['vlogger_entropy']:.4f} bits")

    if args.diversity_lambda > 0.0:
        pub_diff = (div_m['publisher_diversity'] - base_div_m['publisher_diversity']) / (base_div_m['publisher_diversity'] + 1e-8) * 100.0
        vlog_diff = div_m['unique_vloggers'] - base_div_m['unique_vloggers']
        print(f"\nDiversity Improvement vs Baseline:")
        print(f"  - Publisher Diversity Gain:    {pub_diff:+.1f}%  ({base_div_m['publisher_diversity']:.4f} -> {div_m['publisher_diversity']:.4f})")
        print(f"  - Creator Variety Increase:    {vlog_diff:+d} unique vloggers ({base_div_m['unique_vloggers']} -> {div_m['unique_vloggers']})")
    print()


if __name__ == "__main__":
    main()
