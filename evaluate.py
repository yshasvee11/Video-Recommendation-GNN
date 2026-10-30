"""
evaluate.py  –  Vynora 2.0
Evaluation utilities:
  - Accuracy metrics: Recall@K, NDCG@K, Precision@K, MRR@K
  - Paper 5 Diversity metrics: Publisher Diversity@K (Eq 1), Content Diversity@K (Eq 1 & 2),
                               Catalog Coverage@K (Section 2.1)
Supports batched evaluation over all users to avoid OOM on large datasets.
"""

import math
import numpy as np
import torch
import config
from dataloader import VynoraDataset
from model import VynoraGNN
from diversity import diverse_rerank, calculate_diversity_metrics


def evaluate(model: VynoraGNN,
             dataset: VynoraDataset,
             graph_uv: torch.Tensor,
             graph_ua: torch.Tensor,
             mode: str = "test",
             topk_list=None,
             diversity_lambda: float = 0.0,
             diversity_type: str = "publisher",
             diversity_method: str = "mmr",
             compute_diversity: bool = False):
    """
    Full ranking evaluation over all test users, with optional Paper 5 diversity metrics.

    Args:
        model             : trained VynoraGNN (in eval mode)
        dataset           : VynoraDataset instance
        graph_uv          : current (possibly denoised) video-view adjacency
        graph_ua          : current (possibly denoised) vlogger-view adjacency
        mode              : "val" or "test"
        topk_list         : list of K values, e.g. [10, 20]
        diversity_lambda  : float trade-off in [0, 1]. > 0 triggers diverse re-ranking.
        diversity_type    : "publisher" | "content" | "hybrid"
        diversity_method  : "mmr" | "score_rerank"
        compute_diversity : whether to compute Paper 5 diversity metrics (default True if lambda > 0)

    Returns:
        dict of metric_name -> dict of values per K
        e.g. {"Recall": {10: r10, 20: r20}, "NDCG": {10: n10, 20: n20},
              "Publisher_Diversity": {10: d10, 20: d20}, ...}
    """
    if topk_list is None:
        topk_list = config.TOPK_LIST

    target_dict = dataset.testDict if mode == "test" else dataset.valDict
    test_users  = list(target_dict.keys())
    if len(test_users) == 0:
        return {}

    calc_div = compute_diversity or (diversity_lambda > 0.0)

    # Pre-compute vlogger affinity for all videos
    vlogger_of_video = torch.LongTensor(dataset.vlogger_list).to(config.DEVICE)
    vloggers = dataset.vlogger_list

    # Positive videos seen during training (exclude from recommendations)
    allPos_video, _ = dataset.allPos

    model.eval()
    all_recall    = {k: [] for k in topk_list}
    all_ndcg      = {k: [] for k in topk_list}
    all_precision = {k: [] for k in topk_list}
    all_mrr       = {k: [] for k in topk_list}

    # Paper 5 diversity accumulators
    all_pub_div   = {k: [] for k in topk_list}
    all_con_div   = {k: [] for k in topk_list}
    unique_items_recommended = {k: set() for k in topk_list}

    with torch.no_grad():
        # Forward once to get all embeddings
        emb = model.forward(graph_uv, graph_ua)
        user_v    = emb["user_v"]
        video_v   = emb["video_v"]
        user_a    = emb["user_a"]
        vlogger_a = emb["vlogger_a"]

        # Pre-compute vlogger representation for all videos once outside user batches
        p_of_v = vlogger_a[vlogger_of_video]                  # [n_videos, D*(L+1)]

        video_v_np = video_v.detach().cpu().numpy() if calc_div else None
        norm_v_np = None
        if calc_div and video_v_np is not None:
            norms = np.linalg.norm(video_v_np, axis=1, keepdims=True) + 1e-8
            norm_v_np = video_v_np / norms

        max_k = max(topk_list)
        pool_k = max(max_k * 4, 100) if diversity_lambda > 0.0 else max_k
        batch_size = config.TEST_BATCH

        for start in range(0, len(test_users), batch_size):
            batch_users = test_users[start: start + batch_size]
            batch_tensor = torch.LongTensor(batch_users).to(config.DEVICE)

            # Video-view scores  [B, n_videos]
            u_v    = user_v[batch_tensor]
            s_vid  = torch.matmul(u_v, video_v.T)

            # Vlogger-view scores [B, n_videos]
            u_a    = user_a[batch_tensor]
            s_vlog = torch.matmul(u_a, p_of_v.T)             # [B, n_videos]

            scores = (config.LAMBDA_VLOGGER * s_vid
                      + (1 - config.LAMBDA_VLOGGER) * s_vlog) # [B, n_videos]

            # Mask training positives
            for i, user in enumerate(batch_users):
                pos_train = allPos_video[user]
                if len(pos_train) > 0:
                    scores[i, pos_train] = -1e9

            # Retrieve top candidates
            _, candidate_tensor = torch.topk(scores, pool_k, dim=-1)
            candidate_matrix = candidate_tensor.cpu().numpy()
            scores_matrix = scores.cpu().numpy()

            for i, user in enumerate(batch_users):
                gt = target_dict.get(user, None)
                if gt is None:
                    continue

                cands = candidate_matrix[i]
                c_scores = scores_matrix[i, cands]

                # Paper 5 Diverse Re-ranking
                if diversity_lambda > 0.0:
                    ranking = diverse_rerank(
                        candidate_ids=cands,
                        candidate_scores=c_scores,
                        dataset=dataset,
                        topk=max_k,
                        lambda_div=diversity_lambda,
                        diversity_type=diversity_type,
                        method=diversity_method,
                        video_embeddings=norm_v_np
                    )
                else:
                    ranking = cands[:max_k].tolist()

                for k in topk_list:
                    topk_items = ranking[:k]
                    unique_items_recommended[k].update(topk_items)

                    hit        = 1 if gt in topk_items else 0
                    rank       = topk_items.index(gt) + 1 if gt in topk_items else k + 1

                    recall    = hit / 1.0
                    precision = hit / k
                    ndcg      = (math.log(2) / math.log(rank + 1)) if hit else 0.0
                    mrr       = (1.0 / rank) if hit else 0.0

                    all_recall[k].append(recall)
                    all_ndcg[k].append(ndcg)
                    all_precision[k].append(precision)
                    all_mrr[k].append(mrr)

                    if calc_div:
                        # Publisher Diversity (Paper 5 Eq 1)
                        rec_vlogs = [int(vloggers[vid]) for vid in topk_items]
                        counts = {}
                        for vl in rec_vlogs:
                            counts[vl] = counts.get(vl, 0) + 1
                        total_pairs = k * (k - 1)
                        same = sum(c * (c - 1) for c in counts.values() if c > 1)
                        all_pub_div[k].append(1.0 - (same / total_pairs))

                        # Content Diversity (Paper 5 Eq 1 & 2)
                        if norm_v_np is not None:
                            sum_vec = np.sum(norm_v_np[topk_items], axis=0)
                            norm_sq = float(np.dot(sum_vec, sum_vec))
                            diff_sum = total_pairs - (norm_sq - k)
                            all_con_div[k].append(float(np.clip(diff_sum / total_pairs, 0.0, 1.0)))

    results = {}
    for metric_name, metric_dict in [("Recall",    all_recall),
                                     ("NDCG",      all_ndcg),
                                     ("Precision", all_precision),
                                     ("MRR",       all_mrr)]:
        results[metric_name] = {k: float(np.mean(v)) if v else 0.0
                                for k, v in metric_dict.items()}

    if calc_div:
        results["Publisher_Diversity"] = {k: float(np.mean(v)) if v else 0.0
                                         for k, v in all_pub_div.items()}
        if any(len(v) > 0 for v in all_con_div.values()):
            results["Content_Diversity"] = {k: float(np.mean(v)) if v else 0.0
                                           for k, v in all_con_div.items()}
        results["Catalog_Coverage"] = {k: len(unique_items_recommended[k]) / float(dataset.n_videos)
                                      for k in topk_list}

    return results


def format_results(results: dict, topk_list=None) -> str:
    if topk_list is None:
        topk_list = config.TOPK_LIST
    lines = []
    for metric, vals in results.items():
        parts = "  ".join(f"{metric}@{k}={vals[k]:.4f}" for k in topk_list if k in vals)
        if parts:
            lines.append(parts)
    return "\n".join(lines)
