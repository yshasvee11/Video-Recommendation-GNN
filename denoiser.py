"""
denoiser.py  –  Vynora 2.0
CONDE-style edge denoising for the tripartite graph.

CONDE's core denoising loop (from main.py):
  1. After every D epochs, compute a reliability score for each edge
  2. Accumulate: score_all = beta*score_old + (1-beta)*score_new
  3. For each user, drop the lowest-scoring log2(|N(u)|)^gamma * R edges
  4. Rebuild adjacency from retained edges only

This module manages:
  - Tracking raw edge lists (so we can rebuild sparse matrices)
  - Accumulating exponential moving-average reliability scores
  - Applying the edge-drop rule
  - Rebuilding normalised sparse adjacency tensors
"""

import math
import warnings
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
import numpy as np
import torch
import scipy.sparse as sp
import config


def _build_sparse_from_coo(rows, cols, n_users, n_items):
    """
    Build a D^{-½}AD^{-½} normalised bipartite sparse tensor from COO edge lists.
    Returns a torch.sparse_coo_tensor on config.DEVICE.
    """
    n = n_users + n_items
    data = np.ones(len(rows), dtype=np.float32)
    R    = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    # Block adjacency
    adj  = sp.block_array([[None, R], [R.T, None]]).tocoo().astype(np.float32)

    # Degree normalisation
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv  = np.power(rowsum + 1e-12, -0.5)
    D_inv  = sp.diags(d_inv)
    norm   = (D_inv @ adj @ D_inv).tocoo()

    idx = torch.LongTensor(np.vstack([norm.row, norm.col]))
    val = torch.FloatTensor(norm.data)
    sparse = torch.sparse_coo_tensor(idx, val, (n, n)).to(config.DEVICE)
    if config.DEVICE.type == "cpu":
        try:
            sparse = sparse.to_sparse_csr()
        except Exception:
            pass
    return sparse


class GraphDenoiser:
    """
    Manages CONDE-style iterative denoising for one graph view (UV or UA).

    Args:
        train_users : np.ndarray [E]  – source user IDs for each training edge
        train_items : np.ndarray [E]  – target item IDs  (videos or vloggers)
        n_users     : int
        n_items     : int
        beta        : float            – EMA coefficient for score accumulation
        D           : int              – denoising interval (epochs)
        R           : float            – CONDE R parameter (drop fraction)
        gamma       : float            – CONDE gamma parameter (log2 scaling)
        epsilon     : int              – minimum neighbourhood size to trigger dropping
    """

    def __init__(self,
                 train_users: np.ndarray,
                 train_items: np.ndarray,
                 n_users: int,
                 n_items: int,
                 beta:    float = 0.5,
                 D:       int   = 10,
                 R:       float = 0.5,
                 gamma:   float = 1.0,
                 epsilon: int   = 5):
        self.n_users = n_users
        self.n_items = n_items
        self.beta    = beta
        self.D       = D
        self.R       = R
        self.gamma   = gamma
        self.epsilon = epsilon

        self.train_users = train_users.copy()
        self.train_items = train_items.copy()
        self.n_edges     = len(train_users)

        # Edge reliability scores, accumulated across epochs
        self.edge_scores_ema = np.ones(self.n_edges, dtype=np.float32)
        # Boolean mask: True = edge is retained
        self.edge_mask       = np.ones(self.n_edges, dtype=bool)

        # Index: user_id -> list of edge indices (CONDE: trust_index_dict)
        self._user_edge_idx  = self._build_user_edge_index()

        # Initial adjacency
        self._graph = self._rebuild_graph()

    # ────────────────────────────────────────────────────────────────
    def _build_user_edge_index(self):
        idx = [[] for _ in range(self.n_users)]
        for i, u in enumerate(self.train_users):
            idx[u].append(i)
        return idx

    def _rebuild_graph(self):
        retained = self.edge_mask
        rows = self.train_users[retained]
        cols = self.train_items[retained]
        if len(rows) == 0:
            # Safety: if all edges dropped, use full graph
            rows, cols = self.train_users, self.train_items
        return _build_sparse_from_coo(rows, cols, self.n_users, self.n_items)

    @property
    def graph(self):
        return self._graph

    # ────────────────────────────────────────────────────────────────
    def update(self, new_scores: np.ndarray):
        """
        Called after each epoch to accumulate reliability scores.
        Args:
            new_scores: [n_edges] numpy array from model.compute_edge_reliability_*()
        """
        self.edge_scores_ema = (self.beta * self.edge_scores_ema
                                + (1 - self.beta) * new_scores)

    # ────────────────────────────────────────────────────────────────
    def denoise(self):
        """
        Apply CONDE's drop rule: for each user u with |N(u)| >= epsilon,
        drop the bottom log2(|N(u)|)^gamma * R fraction of retained edges.

        Returns: (n_retained, n_dropped)
        """
        dropped = 0
        for u in range(self.n_users):
            edge_ids = self._user_edge_idx[u]
            retained = [e for e in edge_ids if self.edge_mask[e]]
            if len(retained) < self.epsilon:
                continue

            n_drop = int(math.pow(math.log2(max(len(retained), 2)), self.gamma) * self.R)
            if n_drop <= 0:
                continue

            # Sort by EMA score ascending (lowest reliability first)
            scored = sorted(retained, key=lambda e: self.edge_scores_ema[e])
            for e in scored[:n_drop]:
                self.edge_mask[e] = False
            dropped += n_drop

        self._graph = self._rebuild_graph()
        n_retained = int(self.edge_mask.sum())
        return n_retained, dropped

    def stats(self):
        total    = self.n_edges
        retained = int(self.edge_mask.sum())
        return {"total": total, "retained": retained,
                "dropped": total - retained,
                "drop_pct": 100.0 * (total - retained) / total}
