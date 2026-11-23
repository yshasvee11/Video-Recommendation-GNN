"""
denoiser.py – Iterative Edge Pruning & Graph Denoising for Bipartite Graphs (CONDE Adaptation).
Prunes false-positive / noisy interaction edges using learned GNN latent affinity and EMA smoothing.
"""

import numpy as np
import torch
import config


class BipartiteGraphDenoiser:
    """
    Manages edge reliability scoring, temporal EMA accumulation,
    and dynamic quantile edge pruning on the User-Item bipartite graph.
    """
    def __init__(self, train_u, train_i, n_users, n_items, drop_rate=config.DENOISE_DROP_RATE, ema_beta=config.DENOISE_EMA_BETA):
        self.train_u = np.array(train_u, dtype=np.int64)
        self.train_i = np.array(train_i, dtype=np.int64)
        self.n_users = n_users
        self.n_items = n_items
        self.n_edges = len(train_u)
        self.drop_rate = drop_rate
        self.ema_beta = ema_beta
        
        # Reliability scores accumulated across epochs
        self.edge_scores_ema = np.ones(self.n_edges, dtype=np.float32)
        self.active_mask = np.ones(self.n_edges, dtype=bool)

    def update_and_prune(self, model, dataset, chunk_size=65536):
        """
        Computes current representation affinity per edge, updates EMA,
        and rebuilds normalized bipartite adjacency matrix with pruned edges.
        """
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model.get_bipartite_embeddings()
            
            u_t = torch.from_numpy(self.train_u).to(config.DEVICE)
            i_t = torch.from_numpy(self.train_i).to(config.DEVICE)
            
            scores = []
            for s in range(0, self.n_edges, chunk_size):
                e = min(s + chunk_size, self.n_edges)
                sc = torch.sum(user_emb[u_t[s:e]] * item_emb[i_t[s:e]], dim=-1)
                scores.append(sc.cpu().numpy())
            new_scores = np.concatenate(scores)
            
            # EMA Smoothing
            self.edge_scores_ema = self.ema_beta * self.edge_scores_ema + (1.0 - self.ema_beta) * new_scores
            
            # Dynamic Quantile Thresholding (Bottom 5%)
            threshold = np.quantile(self.edge_scores_ema, self.drop_rate)
            self.active_mask = self.edge_scores_ema >= threshold
            
            # Rebuild cleaned bipartite graph
            kept_u = self.train_u[self.active_mask]
            kept_i = self.train_i[self.active_mask]
            pruned_graph = dataset.build_bipartite_adj(kept_u, kept_i)
            
            dropped_count = int(np.sum(~self.active_mask))
            retained_count = int(np.sum(self.active_mask))
            
            return pruned_graph, retained_count, dropped_count
