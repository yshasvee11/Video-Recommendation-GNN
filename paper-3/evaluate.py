"""
evaluate.py – Evaluation Metrics Matching Paper 3 (HetGNN/HHFAN 100-Candidate Protocol & Full Catalog Ranking)
and Genre Diversity Re-Ranking with Lambda Parameter.
"""

import math
import numpy as np
import torch
import config


def compute_genre_diversity(item_ids, genre_list):
    """
    Computes Gini-Simpson genre diversity across recommended items:
    D = 1 - sum(p_g^2)
    """
    n = len(item_ids)
    if n < 2:
        return 1.0
    genres = [genre_list[i] for i in item_ids]
    counts = {}
    for g in genres:
        counts[g] = counts.get(g, 0) + 1
    same_pairs = sum(c * (c - 1) for c in counts.values() if c > 1)
    total_pairs = n * (n - 1)
    return 1.0 - (same_pairs / total_pairs)


def evaluate_model_paper3_protocol(model, dataset, eval_mode="test", topk=config.TOP_K, lambda_div=config.LAMBDA_DIV, sample_users=None):
    """
    Exact Paper 3 (HetGNN / HHFAN / MPGCL) Evaluation Protocol:
    For each test user, their positive items are ranked together with 100 randomly sampled unobserved negative items.
    This matches Table 2 and Table 5 of Paper 3 directly.
    """
    model.eval()
    with torch.no_grad():
        h_u, h_v = model.forward()
        
    target_pos = dataset.test_pos_set if eval_mode == "test" else dataset.val_pos_set
    train_pos = dataset.train_pos_set
    
    users_with_pos = [u for u in range(dataset.n_users) if len(target_pos[u]) > 0]
    if sample_users is not None and sample_users < len(users_with_pos):
        np.random.seed(config.SEED)
        eval_users = np.random.choice(users_with_pos, sample_users, replace=False)
    else:
        eval_users = users_with_pos
        
    precisions, recalls, ndcgs, aucs, divs = [], [], [], [], []
    
    for u in eval_users:
        pos_items = list(target_pos[u])
        train_items = train_pos[u]
        
        # Sample 100 negative unobserved items (HetGNN protocol)
        negs = []
        while len(negs) < 100:
            neg = np.random.randint(0, dataset.n_items)
            if neg not in train_items and neg not in target_pos[u]:
                negs.append(neg)
                
        cands = pos_items + negs
        c_vecs = h_v[cands]
        u_vec = h_u[u]
        scores = torch.mv(c_vecs, u_vec).cpu().numpy()
        
        # AUC on positives vs 100 negatives
        pos_scores = scores[:len(pos_items)]
        neg_scores = scores[len(pos_items):]
        auc = np.mean(pos_scores[:, None] > neg_scores[None, :]) + 0.5 * np.mean(pos_scores[:, None] == neg_scores[None, :])
        aucs.append(auc)
        
        # Rank candidates with optional lambda genre diversification
        if lambda_div <= 0.0:
            top_cand_idx = np.argpartition(-scores, topk)[:topk]
            top_ranked = [cands[i] for i in top_cand_idx[np.argsort(-scores[top_cand_idx])]]
        else:
            # MMR Diversity Re-ranking over candidate pool
            norm_rel = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            selected = []
            selected_g = set()
            rem = list(range(len(cands)))
            for _ in range(topk):
                best_r = -1
                best_mmr = -1e9
                for r in rem:
                    g = dataset.genre_list[cands[r]]
                    dist = 1.0 if g not in selected_g else 0.0
                    mmr = (1.0 - lambda_div) * norm_rel[r] + lambda_div * dist
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_r = r
                selected.append(cands[best_r])
                selected_g.add(dataset.genre_list[cands[best_r]])
                rem.remove(best_r)
            top_ranked = selected
            
        hits = [1 if item in target_pos[u] else 0 for item in top_ranked]
        precisions.append(sum(hits) / topk)
        recalls.append(sum(hits) / len(pos_items))
        dcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(hits))
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(topk, len(pos_items))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        divs.append(compute_genre_diversity(top_ranked, dataset.genre_list))
        
    return {
        "Precision@10": float(np.mean(precisions)),
        "Recall@10": float(np.mean(recalls)),
        "NDCG@10": float(np.mean(ndcgs)),
        "AUC": float(np.mean(aucs)),
        "Genre_Diversity": float(np.mean(divs)),
        "Evaluated_Users": len(eval_users)
    }


def evaluate_model_full_catalog(model, dataset, eval_mode="test", topk=config.TOP_K, lambda_div=config.LAMBDA_DIV, sample_users=1000):
    """
    Full Catalog Evaluation across all 3,706 items.
    """
    model.eval()
    with torch.no_grad():
        h_u, h_v = model.forward()
        
    target_pos = dataset.test_pos_set if eval_mode == "test" else dataset.val_pos_set
    train_pos = dataset.train_pos_set
    
    test_users = [u for u in range(dataset.n_users) if len(target_pos[u]) > 0]
    if sample_users is not None and sample_users < len(test_users):
        np.random.seed(config.SEED)
        test_users = np.random.choice(test_users, sample_users, replace=False)
        
    precisions, recalls, ndcgs, aucs = [], [], [], []
    h_v_t = h_v.t()
    
    for u in test_users:
        pos_items = target_pos[u]
        train_items = train_pos[u]
        scores = torch.mm(h_u[u].unsqueeze(0), h_v_t).squeeze(0).cpu().numpy()
        scores[list(train_items)] = -1e9
        
        # Global AUC
        pos_mask = np.zeros(len(scores), dtype=bool)
        pos_mask[list(pos_items)] = True
        neg_mask = np.ones(len(scores), dtype=bool)
        neg_mask[list(train_items)] = False
        neg_mask[list(pos_items)] = False
        pos_sc = scores[pos_mask]
        neg_sc = scores[neg_mask]
        if len(pos_sc) > 0 and len(neg_sc) > 0:
            neg_sample = np.random.choice(neg_sc, min(1000, len(neg_sc)), replace=False)
            aucs.append(np.mean(pos_sc[:, None] > neg_sample[None, :]))
            
        top_idx = np.argpartition(-scores, topk)[:topk]
        top_ranked = top_idx[np.argsort(-scores[top_idx])]
        
        hits = [1 if item in pos_items else 0 for item in top_ranked]
        precisions.append(sum(hits) / topk)
        recalls.append(sum(hits) / len(pos_items))
        dcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(hits))
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(topk, len(pos_items))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
    return {
        "Precision@10": float(np.mean(precisions)),
        "Recall@10": float(np.mean(recalls)),
        "NDCG@10": float(np.mean(ndcgs)),
        "AUC": float(np.mean(aucs)) if len(aucs) > 0 else 0.0,
        "Evaluated_Users": len(test_users)
    }

# Default evaluation function points to Paper 3 protocol for exact benchmark comparison
evaluate_model = evaluate_model_paper3_protocol
