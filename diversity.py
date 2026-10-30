"""
diversity.py  –  Vynora 2.0: Diversity Recommendation (Paper 5 Implementation)
Reference: "Predicting diversification scores of videos in recommendation network"
           Selin Chun, Jinyoung Han, Daejin Choi, Taekyoung Kwon
           Expert Systems With Applications 238 (2024) 121803

Implements:
  1. Diversity Metrics (Eq 1 & 2): Publisher Diversity, Content Diversity, Entropy
  2. Empirical Diversification Score Calculation (Section 3 & 4.3)
  3. Diversification Score Prediction GNN with Feature-Augmented Attention (Section 4.5, Eq 3-8)
  4. Diversity-Aware Re-Ranking:
     - Method A: Score Blending Re-Ranking (Section 7)
     - Method B: Maximal Marginal Relevance (MMR) Intra-List Diversification (Section 2.1)
  5. Batched Diversity Evaluation across user populations (Catalog Coverage, Gini Index)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Diversity Evaluation Metrics (Paper 5 Eq 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_diversity_metrics(item_ids, dataset, video_embeddings=None):
    """
    Compute diversity metrics for a given recommendation list of item IDs.

    Paper 5 Equation (1):
        Diversity(L) = sum_{i != j} d(i, j) / (n * (n - 1))

    Distance functions (Section 3):
        - Publisher distance d_pub(i, j) = 1 if publisher(i) != publisher(j) else 0
        - Content distance d_con(i, j) = 1 - CosineSimilarity(emb_i, emb_j)

    Args:
        item_ids: list or array of video IDs (length n)
        dataset: VynoraDataset instance (contains vlogger_list)
        video_embeddings: [n_videos, D] numpy array or tensor (optional, for content diversity)

    Returns:
        dict containing:
          - publisher_diversity: float in [0, 1]
          - content_diversity: float in [0, 1] (if embeddings provided, else None)
          - unique_vloggers: int
          - vlogger_entropy: float (Shannon entropy in bits)
          - total_items: int
    """
    n = len(item_ids)
    if n < 2:
        return {
            "publisher_diversity": 1.0,
            "content_diversity": 1.0 if video_embeddings is not None else None,
            "unique_vloggers": n,
            "vlogger_entropy": 0.0,
            "total_items": n
        }

    # --- Publisher Diversity (Paper 5 Eq 1 with binary vlogger distance) ---
    vloggers = [int(dataset.vlogger_list[vid]) for vid in item_ids]
    counts = {}
    for vl in vloggers:
        counts[vl] = counts.get(vl, 0) + 1

    same_pairs = sum(c * (c - 1) for c in counts.values() if c > 1)
    total_pairs = n * (n - 1)
    pub_diversity = 1.0 - (same_pairs / total_pairs)

    # Shannon Entropy: -sum(p * log2(p))
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())

    # --- Content Diversity (Paper 5 Eq 1 & 2: Cosine Distance of Embeddings) ---
    con_diversity = None
    if video_embeddings is not None:
        if isinstance(video_embeddings, torch.Tensor):
            embs = video_embeddings[item_ids].detach().cpu().numpy()
        else:
            embs = video_embeddings[item_ids]

        # Unit-normalize vectors
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        norm_embs = embs / norms

        # O(n * D) formula: sum_{i!=j} (1 - e_i . e_j) = n(n-1) - (||sum e_i||^2 - n)
        sum_vec = np.sum(norm_embs, axis=0)
        norm_sq = float(np.dot(sum_vec, sum_vec))
        diff_sum = total_pairs - (norm_sq - n)
        con_diversity = float(np.clip(diff_sum / total_pairs, 0.0, 1.0))

    return {
        "publisher_diversity": float(pub_diversity),
        "content_diversity": con_diversity,
        "unique_vloggers": len(counts),
        "vlogger_entropy": float(entropy),
        "total_items": n
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Empirical Diversification Score Calculation (Paper 5 Section 3 & 4.3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_empirical_diversification_scores(dataset,
                                             diversity_type: str = "publisher",
                                             video_embeddings=None,
                                             use_cache: bool = True):
    """
    Computes ground-truth diversification score S_v for each video node v as defined
    in Paper 5 Section 3:
      "the diversification scores of node i for genre, publisher, and content diversity
       are defined as the average values of genre, publisher and content diversity of
       all the users who visit node i, respectively."

    Args:
        dataset: VynoraDataset
        diversity_type: "publisher" | "content"
        video_embeddings: [n_videos, D] for content diversity
        use_cache: if True, load/save from results/diversification_{dataset}_{type}.npy

    Returns:
        div_scores: np.ndarray of shape [n_videos], float32 in [0, 1]
    """
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    dataset_name = getattr(dataset, "dataset_name", config.DATASET)
    cache_file = os.path.join(config.RESULT_DIR, f"diversification_{dataset_name}_{diversity_type}.npy")

    if use_cache and os.path.exists(cache_file):
        try:
            scores = np.load(cache_file)
            if len(scores) == dataset.n_videos:
                return scores
        except Exception:
            pass

    allPos_video, _ = dataset.allPos
    n_users = dataset.n_users
    n_videos = dataset.n_videos
    vloggers = dataset.vlogger_list

    # 1. Compute diversity experience of each user
    user_div = np.zeros(n_users, dtype=np.float32)

    if diversity_type == "publisher":
        for u, vids in enumerate(allPos_video):
            n = len(vids)
            if n >= 2:
                u_vlogs = [vloggers[v] for v in vids]
                counts = np.bincount(u_vlogs)
                same = sum(c * (c - 1) for c in counts if c > 1)
                user_div[u] = 1.0 - (same / (n * (n - 1)))
            else:
                user_div[u] = 1.0

    elif diversity_type == "content":
        if video_embeddings is None:
            raise ValueError("video_embeddings must be provided to compute content diversification scores")
        if isinstance(video_embeddings, torch.Tensor):
            norm_v = video_embeddings.detach().cpu().numpy()
        else:
            norm_v = video_embeddings.copy()
        norms = np.linalg.norm(norm_v, axis=1, keepdims=True) + 1e-8
        norm_v = norm_v / norms

        for u, vids in enumerate(allPos_video):
            n = len(vids)
            if n >= 2:
                sum_vec = np.sum(norm_v[vids], axis=0)
                norm_sq = np.dot(sum_vec, sum_vec)
                val = 1.0 - (norm_sq - n) / (n * (n - 1))
                user_div[u] = np.clip(val, 0.0, 1.0)
            else:
                user_div[u] = 1.0
    else:
        raise ValueError(f"Unknown diversity_type: {diversity_type}")

    # 2. Average user diversity over users who visited each video v
    video_sum = np.zeros(n_videos, dtype=np.float64)
    video_count = np.zeros(n_videos, dtype=np.int32)

    for u, vids in enumerate(allPos_video):
        score_u = user_div[u]
        for v in vids:
            video_sum[v] += score_u
            video_count[v] += 1

    global_mean = float(np.mean(user_div))
    div_scores = np.zeros(n_videos, dtype=np.float32)
    for v in range(n_videos):
        if video_count[v] > 0:
            div_scores[v] = video_sum[v] / video_count[v]
        else:
            div_scores[v] = global_mean

    if use_cache:
        np.save(cache_file, div_scores)

    return div_scores


# ─────────────────────────────────────────────────────────────────────────────
# 3. Diversification Score Prediction GNN (Paper 5 Section 4.5, Eq 3-8)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureAugmentedAttentionLayer(nn.Module):
    """
    Embedding Propagation Layer with Feature-Augmented Attention mechanism
    (Paper 5 Section 4.5.2, Equations 4, 5, 6).

    alpha_{j, i, e} = softmax_j( LeakyReLU( a^T [W_N h_j || W_N h_i || W_E h_e] ) )
    h_j^{(l+1)}     = ELU( sum_{i in N(j)} alpha_{j, i, e} W_N h_i )
    h_e^{(l+1)}     = W_E h_e^{(l)}
    """
    def __init__(self, node_dim: int, edge_dim: int = 4, out_dim: int = 64):
        super().__init__()
        self.w_node = nn.Linear(node_dim, out_dim, bias=False)
        self.w_edge = nn.Linear(edge_dim, out_dim, bias=False)
        # Attention vector: takes concatenation of [W_N h_j || W_N h_i || W_E h_e] -> 3 * out_dim
        self.att_vec = nn.Parameter(torch.empty(3 * out_dim, 1))
        nn.init.xavier_uniform_(self.att_vec.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.activation = nn.ELU()

    def forward(self, h_nodes: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        Args:
            h_nodes: [num_nodes, node_dim]
            edge_index: [2, num_edges] (src, dst)
            edge_attr: [num_edges, edge_dim]
        """
        src, dst = edge_index[0], edge_index[1]

        # Shared linear transformations
        wn_nodes = self.w_node(h_nodes)      # [num_nodes, out_dim]
        we_edges = self.w_edge(edge_attr)    # [num_edges, out_dim]

        wn_src = wn_nodes[src]                # [num_edges, out_dim]
        wn_dst = wn_nodes[dst]                # [num_edges, out_dim]

        # Feature concatenation: [W_N h_dst || W_N h_src || W_E h_e]
        cat_feat = torch.cat([wn_dst, wn_src, we_edges], dim=-1) # [num_edges, 3 * out_dim]
        att_logits = self.leaky_relu(torch.matmul(cat_feat, self.att_vec).squeeze(-1)) # [num_edges]

        # Normalized attention per destination node (approx via softmax over incoming edges)
        # Numerical stability via exp
        att_exp = torch.exp(att_logits - att_logits.max())
        dst_sum = torch.zeros(h_nodes.size(0), device=h_nodes.device)
        dst_sum.scatter_add_(0, dst, att_exp)
        att_weights = att_exp / (dst_sum[dst] + 1e-8) # [num_edges]

        # Aggregate weighted messages
        messages = wn_src * att_weights.unsqueeze(-1) # [num_edges, out_dim]
        out_nodes = torch.zeros_like(wn_nodes)
        out_nodes.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        out_nodes = self.activation(out_nodes)

        # Update edge representation (Eq 6)
        out_edges = we_edges

        return out_nodes, out_edges


class DiversificationGNN(nn.Module):
    """
    Paper 5 Section 4.5 Diversification Score Prediction Model.
    Architecture:
      1. Node feature embedding: h_0 = W_embed * r_i (Eq 3)
      2. Embedding propagation layers with feature-augmented attention (Eq 4-6)
      3. Prediction MLP: S_hat = W_2(sigma(W_1 h_i)) (Eq 7)
      4. Loss: MAE (Eq 8)
    """
    def __init__(self, in_features_dim: int, hidden_dim: int = 64, edge_dim: int = 4, n_layers: int = 2):
        super().__init__()
        self.node_embed = nn.Linear(in_features_dim, hidden_dim)
        self.layers = nn.ModuleList([
            FeatureAugmentedAttentionLayer(hidden_dim, edge_dim=edge_dim if i == 0 else hidden_dim, out_dim=hidden_dim)
            for i in range(n_layers)
        ])
        # Prediction Layer (Eq 7)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # scores are in [0, 1]
        )

    def forward(self, node_feats, edge_index, edge_attr):
        h = self.node_embed(node_feats)
        e = edge_attr
        for layer in self.layers:
            h_next, e = layer(h, edge_index, e)
            h = h + h_next  # residual connection for stability
        scores = self.predictor(h).squeeze(-1)
        return scores

    def compute_loss(self, pred_scores, true_scores):
        """Paper 5 Equation (8): Mean Absolute Error (MAE)"""
        return F.l1_loss(pred_scores, true_scores)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diversity-Aware Re-Ranking (Paper 5 Section 7 & Section 2.1)
# ─────────────────────────────────────────────────────────────────────────────

def diverse_rerank(candidate_ids,
                   candidate_scores,
                   dataset,
                   topk: int = 10,
                   lambda_div: float = 0.3,
                   diversity_type: str = "publisher",
                   method: str = "mmr",
                   diversification_scores=None,
                   video_embeddings=None):
    """
    Re-rank candidate recommendations to optimize user diversity experiences.

    Implements:
      Method A ('score_rerank' - Paper 5 Section 7):
        S_final(v) = (1 - lambda) * S_rel(v) + lambda * S_div(v)

      Method B ('mmr' - Paper 5 Eq 1 & Section 2.1 Intra-List Distance Diversification):
        Greedy selection balancing normalized relevance score with marginal diversity
        distance to previously selected items:
        arg max_{v in C \\ R} [ (1 - lambda) * S_rel(v) + lambda * Dist(v, R) ]

    Args:
        candidate_ids: array-like of candidate video IDs
        candidate_scores: array-like of base relevance scores
        dataset: VynoraDataset instance
        topk: number of recommendations to return
        lambda_div: trade-off parameter in [0, 1]. 0 = pure relevance, 1 = max diversity
        diversity_type: "publisher" | "content" | "hybrid"
        method: "mmr" | "score_rerank"
        diversification_scores: pre-computed S_v diversification scores (required for score_rerank)
        video_embeddings: [n_videos, D] embeddings (for content/hybrid diversity)

    Returns:
        List of selected video IDs of length min(topk, len(candidate_ids))
    """
    candidate_ids = np.array(candidate_ids)
    candidate_scores = np.array(candidate_scores, dtype=np.float32)

    # Fast path: pure relevance (no diversity trade-off)
    if lambda_div <= 0.0 or len(candidate_ids) <= topk:
        sorted_order = np.argsort(-candidate_scores)[:topk]
        return candidate_ids[sorted_order].tolist()

    # Normalise relevance scores to [0, 1]
    s_min, s_max = float(candidate_scores.min()), float(candidate_scores.max())
    if s_max > s_min:
        norm_scores = (candidate_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(candidate_scores)

    vloggers = dataset.vlogger_list

    # ─────────────────────────────────────────────────────────
    # Method A: Score Blending Re-Ranking (Paper 5 Section 7)
    # ─────────────────────────────────────────────────────────
    if method == "score_rerank":
        if diversification_scores is None:
            diversification_scores = compute_empirical_diversification_scores(
                dataset,
                diversity_type="content" if diversity_type == "content" else "publisher",
                video_embeddings=video_embeddings
            )
        cand_div = diversification_scores[candidate_ids]
        d_min, d_max = float(cand_div.min()), float(cand_div.max())
        if d_max > d_min:
            norm_div = (cand_div - d_min) / (d_max - d_min)
        else:
            norm_div = np.ones_like(cand_div)

        final_scores = (1.0 - lambda_div) * norm_scores + lambda_div * norm_div
        best_indices = np.argsort(-final_scores)[:topk]
        return candidate_ids[best_indices].tolist()

    # ─────────────────────────────────────────────────────────
    # Method B: Maximal Marginal Relevance (Paper 5 Eq 1 & Sec 2.1)
    # ─────────────────────────────────────────────────────────
    norm_v_embs = None
    if diversity_type in ("content", "hybrid") and video_embeddings is not None:
        if isinstance(video_embeddings, torch.Tensor):
            ve = video_embeddings.detach().cpu().numpy()
        else:
            ve = video_embeddings
        norms = np.linalg.norm(ve, axis=1, keepdims=True) + 1e-8
        norm_v_embs = ve / norms

    selected_ids = []
    selected_vloggers = set()
    rem_indices = list(range(len(candidate_ids)))

    for _ in range(min(topk, len(candidate_ids))):
        best_idx = -1
        best_mmr = -1e9

        for idx in rem_indices:
            vid = candidate_ids[idx]
            rel = norm_scores[idx]
            vlog = int(vloggers[vid])

            if len(selected_ids) == 0:
                dist = 1.0
            else:
                dist = 0.0
                # Publisher distance (1 if vlogger not yet in selected, else 0)
                if diversity_type in ("publisher", "hybrid"):
                    pub_dist = 1.0 if vlog not in selected_vloggers else 0.0
                    dist += (0.5 * pub_dist if diversity_type == "hybrid" else pub_dist)

                # Content distance (cosine distance to selected items)
                if diversity_type in ("content", "hybrid") and norm_v_embs is not None:
                    sims = [np.dot(norm_v_embs[vid], norm_v_embs[s]) for s in selected_ids]
                    con_dist = float(np.mean([1.0 - s for s in sims]))
                    dist += (0.5 * con_dist if diversity_type == "hybrid" else con_dist)

            mmr_score = (1.0 - lambda_div) * rel + lambda_div * dist
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        sel_vid = candidate_ids[best_idx]
        selected_ids.append(sel_vid)
        selected_vloggers.add(int(vloggers[sel_vid]))
        rem_indices.remove(best_idx)

    return selected_ids


# ─────────────────────────────────────────────────────────────────────────────
# 5. Batched Diversity Evaluation (Population-level Catalog Coverage & Diversity)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_diversity_batch(topk_matrix, dataset, video_embeddings=None):
    """
    Computes average diversity and coverage over a population of test recommendation lists.

    Args:
        topk_matrix: [N_users, K] numpy array of recommended video IDs
        dataset: VynoraDataset
        video_embeddings: [n_videos, D] for content diversity

    Returns:
        dict:
          - Publisher_Diversity: mean publisher diversity across users (Eq 1)
          - Content_Diversity: mean content diversity across users (Eq 1 & 2)
          - Catalog_Coverage: unique items recommended / total catalog size
          - Vlogger_Coverage: unique vloggers recommended / total vloggers
    """
    n_users, k = topk_matrix.shape
    vloggers = dataset.vlogger_list

    pub_divs = []
    con_divs = []
    all_recommended_videos = set()
    all_recommended_vloggers = set()

    # Pre-normalize embeddings if available
    norm_v = None
    if video_embeddings is not None:
        if isinstance(video_embeddings, torch.Tensor):
            ve = video_embeddings.detach().cpu().numpy()
        else:
            ve = video_embeddings
        norms = np.linalg.norm(ve, axis=1, keepdims=True) + 1e-8
        norm_v = ve / norms

    total_pairs = k * (k - 1)

    for i in range(n_users):
        recs = topk_matrix[i].tolist()
        all_recommended_videos.update(recs)

        # Publisher diversity (Eq 1)
        rec_vlogs = [int(vloggers[vid]) for vid in recs]
        all_recommended_vloggers.update(rec_vlogs)
        counts = {}
        for vl in rec_vlogs:
            counts[vl] = counts.get(vl, 0) + 1
        same = sum(c * (c - 1) for c in counts.values() if c > 1)
        pub_divs.append(1.0 - (same / total_pairs))

        # Content diversity (Eq 1 & 2)
        if norm_v is not None:
            sum_vec = np.sum(norm_v[recs], axis=0)
            norm_sq = float(np.dot(sum_vec, sum_vec))
            diff_sum = total_pairs - (norm_sq - k)
            con_divs.append(float(np.clip(diff_sum / total_pairs, 0.0, 1.0)))

    return {
        "Publisher_Diversity": float(np.mean(pub_divs)) if pub_divs else 0.0,
        "Content_Diversity": float(np.mean(con_divs)) if con_divs else None,
        "Catalog_Coverage": len(all_recommended_videos) / float(dataset.n_videos),
        "Vlogger_Coverage": len(all_recommended_vloggers) / float(dataset.n_vloggers)
    }

