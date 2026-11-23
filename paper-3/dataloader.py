"""
dataloader.py – MovieLens-1M Dataset Loader with 80:10:10 Random Split & Meta-Path Homogeneous Graphs.
Follows Paper 3 (MPGCL, ESWA 2023) experimental setup.
"""

import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import config

try:
    torch.sparse.check_sparse_tensor_invariants.disable()
except Exception:
    pass


def _normalize_sparse_adj(adj):
    """
    Symmetric Laplacian degree normalization: D^{-1/2} A D^{-1/2}
    """
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.zeros_like(rowsum, dtype=np.float32)
    pos_mask = rowsum > 0
    d_inv[pos_mask] = np.power(rowsum[pos_mask], -0.5)
    d_mat = sp.diags(d_inv)
    norm_adj = d_mat.dot(adj).dot(d_mat).tocoo()
    
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col))).long()
    values = torch.from_numpy(norm_adj.data).float()
    shape = torch.Size(norm_adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce().to(config.DEVICE)


def _filter_topk_sparse(mat, k=10):
    """
    Keep top-k strongest neighbor connections per row in a sparse matrix.
    """
    mat = mat.tocsr()
    new_rows, new_cols, new_data = [], [], []
    for r in range(mat.shape[0]):
        row_data = mat.data[mat.indptr[r]:mat.indptr[r+1]]
        row_cols = mat.indices[mat.indptr[r]:mat.indptr[r+1]]
        if len(row_data) > k:
            top_idx = np.argpartition(-row_data, k)[:k]
            new_rows.extend([r] * k)
            new_cols.extend(row_cols[top_idx])
            new_data.extend(row_data[top_idx])
        elif len(row_data) > 0:
            new_rows.extend([r] * len(row_data))
            new_cols.extend(row_cols)
            new_data.extend(row_data)
    return sp.coo_matrix((new_data, (new_rows, new_cols)), shape=mat.shape)


class MovieLensDataset:
    """
    MovieLens-1M Dataset loader implementing Paper 3's bipartite and meta-path homogeneous representations.
    """
    def __init__(self, data_dir=config.DATA_DIR, seed=config.SEED):
        self.data_dir = data_dir
        self.seed = seed
        
        # 1. Load Ratings
        ratings_path = os.path.join(data_dir, "ratings.dat")
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"MovieLens ratings.dat not found at: {ratings_path}")
            
        ratings = pd.read_csv(
            ratings_path, sep="::", engine="python",
            names=["u_id", "m_id", "rating", "timestamp"]
        )
        
        # 2. Contiguous ID Mappings
        unique_users = sorted(ratings["u_id"].unique())
        unique_movies = sorted(ratings["m_id"].unique())
        
        self.u_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.i_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.inv_u_map = {idx: uid for uid, idx in self.u_map.items()}
        self.inv_i_map = {idx: mid for mid, idx in self.i_map.items()}
        
        self.n_users = len(self.u_map)    # 6,040
        self.n_items = len(self.i_map)    # 3,706
        self.n_interactions = len(ratings) # 1,000,209
        
        u_indices = ratings["u_id"].map(self.u_map).values
        i_indices = ratings["m_id"].map(self.i_map).values
        
        # 3. Load Genres for Diversity Evaluation
        movies_path = os.path.join(data_dir, "movies.dat")
        self.genre_list = [0] * self.n_items
        self.genre_names = []
        if os.path.exists(movies_path):
            movies = pd.read_csv(
                movies_path, sep="::", engine="python",
                names=["m_id", "title", "genres"], encoding="latin-1"
            )
            all_genres = sorted(list(set(g for gs in movies["genres"] for g in gs.split("|"))))
            self.genre_names = all_genres
            genre_to_id = {g: idx for idx, g in enumerate(all_genres)}
            
            for _, row in movies.iterrows():
                mid = row["m_id"]
                if mid in self.i_map:
                    item_idx = self.i_map[mid]
                    # Assign primary genre (first listed genre)
                    primary_g = row["genres"].split("|")[0]
                    self.genre_list[item_idx] = genre_to_id.get(primary_g, 0)
                    
        # 4. Random 80:10:10 Train / Val / Test Split (Matching Paper 3 Table 5)
        np.random.seed(self.seed)
        shuffled_indices = np.random.permutation(self.n_interactions)
        
        n_train = int(0.8 * self.n_interactions)
        n_val = int(0.1 * self.n_interactions)
        
        train_idx = shuffled_indices[:n_train]
        val_idx = shuffled_indices[n_train:n_train + n_val]
        test_idx = shuffled_indices[n_train + n_val:]
        
        self.train_u = u_indices[train_idx]
        self.train_i = i_indices[train_idx]
        
        self.val_u = u_indices[val_idx]
        self.val_i = i_indices[val_idx]
        
        self.test_u = u_indices[test_idx]
        self.test_i = i_indices[test_idx]
        
        # User positive sets
        self.train_pos_set = {u: set() for u in range(self.n_users)}
        for u, i in zip(self.train_u, self.train_i):
            self.train_pos_set[u].add(i)
            
        self.val_pos_set = {u: set() for u in range(self.n_users)}
        for u, i in zip(self.val_u, self.val_i):
            self.val_pos_set[u].add(i)
            
        self.test_pos_set = {u: set() for u in range(self.n_users)}
        for u, i in zip(self.test_u, self.test_i):
            self.test_pos_set[u].add(i)
            
        print(f"Loaded MovieLens-1M:")
        print(f"  Users: {self.n_users:,} | Items: {self.n_items:,} | Total Ratings: {self.n_interactions:,}")
        print(f"  Train: {len(self.train_u):,} (80%) | Val: {len(self.val_u):,} (10%) | Test: {len(self.test_u):,} (10%)")
        
        # 5. Build Bipartite Adjacency Matrix
        self.graph_ui = self.build_bipartite_adj(self.train_u, self.train_i)
        
        # 6. Build Meta-Path Homogeneous Graphs (U-I-U and I-U-I)
        self.meta_graph_u, self.meta_graph_i = self.build_metapath_homogeneous_graphs()

    def build_bipartite_adj(self, u_list, i_list):
        """
        Builds (N_u + N_i) x (N_u + N_i) normalized sparse bipartite adjacency.
        """
        n_total = self.n_users + self.n_items
        rows = np.concatenate([u_list, i_list + self.n_users])
        cols = np.concatenate([i_list + self.n_users, u_list])
        vals = np.ones(len(rows), dtype=np.float32)
        
        adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_total, n_total))
        return _normalize_sparse_adj(adj)

    def build_metapath_homogeneous_graphs(self, top_k=config.META_TOP_K):
        """
        Constructs User-User (U-I-U) and Item-Item (I-U-I) homogeneous graphs
        from meta-path co-occurrences.
        """
        R = sp.csr_matrix(
            (np.ones(len(self.train_u), dtype=np.float32), (self.train_u, self.train_i)),
            shape=(self.n_users, self.n_items)
        )
        
        # Item-Item Meta-Path (V-U-V)
        S_v = (R.T @ R).tocsr()
        S_v.setdiag(0)
        S_v.eliminate_zeros()
        S_v_topk = _filter_topk_sparse(S_v, k=top_k)
        norm_S_v = _normalize_sparse_adj(S_v_topk)
        
        # User-User Meta-Path (U-V-U)
        S_u = (R @ R.T).tocsr()
        S_u.setdiag(0)
        S_u.eliminate_zeros()
        S_u_topk = _filter_topk_sparse(S_u, k=top_k)
        norm_S_u = _normalize_sparse_adj(S_u_topk)
        
        return norm_S_u, norm_S_v
