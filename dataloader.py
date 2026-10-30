"""
dataloader.py  –  Vynora 2.0
Loads the tripartite VA-GNN-format datasets (takatak / wechat).

Data format (same as original VA-GNN):
  user_video_data.txt   : each line -> "user_id  v1 v2 ... vN"   (train/val/test split: last 2 are val+test)
  user_vlogger_data.txt : each line -> "user_id  a1 a2 ... aN"   (train/val/test split)
  user_vlogger_data2.txt: extra user-vlogger supervision (last 2 for val+test)
  vlogger_video_data.txt: tab-sep pairs "vlogger_id  video_id"
  data_size.txt         : "n_users<tab>n_videos<tab>n_vloggers"
"""

import warnings
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
import os
import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
import config


class VynoraDataset(Dataset):
    """
    Full tripartite dataset for Vynora 2.0.
    Exposes:
      - Sparse adjacency matrices for both graph views
      - allPos / allPos2 for BPR sampling
      - vlogger_list[video_id] -> vlogger_id
    """

    def __init__(self, dataset_name: str = config.DATASET):
        path = os.path.join(config.DATA_ROOT, dataset_name)
        print(f"[VynoraDataset] Loading '{dataset_name}' from {path}")

        # ── read sizes ──────────────────────────────────────────────────
        with open(os.path.join(path, "data_size.txt")) as f:
            parts = f.readline().strip().split("\t")
        self.n_users, self.n_videos, self.n_vloggers = int(parts[0]), int(parts[1]), int(parts[2])
        print(f"  users={self.n_users}  videos={self.n_videos}  vloggers={self.n_vloggers}")

        # ── vlogger_list: video_id -> vlogger_id ───────────────────────
        self.vlogger_list = np.zeros(self.n_videos, dtype=np.int64)
        with open(os.path.join(path, "vlogger_video_data.txt")) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.vlogger_list[int(parts[1])] = int(parts[0])

        # ── user-video interactions ─────────────────────────────────────
        (self._train_uv_users, self._train_uv_videos,
         self._val_uv,  self._test_uv,
         self._allPos_video) = self._load_interactions(
            os.path.join(path, "user_video_data.txt"),
            self.n_users, self.n_videos
        )
        self.trainDataSize = len(self._train_uv_users)

        # ── user-vlogger interactions (primary supervision) ─────────────
        (self._train_ua_users, self._train_ua_vloggers,
         self._val_ua, self._test_ua,
         self._allPos_vlogger) = self._load_interactions(
            os.path.join(path, "user_vlogger_data.txt"),
            self.n_users, self.n_vloggers
        )
        self.trainDataSize2 = len(self._train_ua_users)

        # ── user-vlogger extra data (for allPos2 used in BPR sampling) ──
        self._allPos2 = self._load_allpos_only(
            os.path.join(path, "user_vlogger_data2.txt"),
            self.n_users
        )

        # ── precompute sets for fast O(1) sampling ───────────────────────
        self._allPos_video_set = [set(v) for v in self._allPos_video]
        self._allPos_vlogger_set = [set(a) for a in self._allPos_vlogger]
        self._allPos2_set = [set(a) for a in self._allPos2]

        # ── build sparse graph matrices ─────────────────────────────────
        self._graph_uv = None   # lazy build
        self._graph_ua = None

        print(f"  train U-V pairs={self.trainDataSize}  train U-A pairs={self.trainDataSize2}")
        print("[VynoraDataset] Done.\n")

    # ────────────────────────────────────────────────────────────────────
    # Public properties
    # ────────────────────────────────────────────────────────────────────

    @property
    def allPos(self):
        """Returns (allPos_video, allPos_vlogger) dicts: user -> list of positives."""
        return self._allPos_video, self._allPos_vlogger

    @property
    def allPosSets(self):
        """Returns (allPos_video_set, allPos_vlogger_set): user -> set of positives."""
        return self._allPos_video_set, self._allPos_vlogger_set

    @property
    def allPos2(self):
        """Extra user-vlogger positives (from user_vlogger_data2)."""
        return self._allPos2

    @property
    def allPos2Sets(self):
        """Extra user-vlogger positive sets."""
        return self._allPos2_set

    @property
    def valDict(self):
        return self._val_uv

    @property
    def testDict(self):
        return self._test_uv

    # ────────────────────────────────────────────────────────────────────
    # Sparse adjacency (video-view and vlogger-view)
    # ────────────────────────────────────────────────────────────────────

    def get_sparse_graph_uv(self):
        """
        Normalised bipartite User-Video adjacency as torch sparse tensor.
        Block form:
            A = |0   R  |
                |R^T 0  |
        size: (n_users + n_videos) x (n_users + n_videos)
        """
        if self._graph_uv is None:
            self._graph_uv = self._build_bipartite_sparse(
                self._train_uv_users, self._train_uv_videos,
                self.n_users, self.n_videos
            )
        return self._graph_uv

    def get_sparse_graph_ua(self):
        """Normalised bipartite User-Vlogger adjacency."""
        if self._graph_ua is None:
            self._graph_ua = self._build_bipartite_sparse(
                self._train_ua_users, self._train_ua_vloggers,
                self.n_users, self.n_vloggers
            )
        return self._graph_ua

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_interactions(filepath, n_users, n_items):
        """
        Parse VA-GNN-format interaction file.
        Returns:
          train_users, train_items  (numpy arrays, paired)
          val_dict, test_dict       (user -> item id)
          allPos                    (user -> list of train item ids)
        """
        train_users, train_items = [], []
        val_dict  = {}
        test_dict = {}
        allPos    = [[] for _ in range(n_users)]

        with open(filepath) as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                uid = int(tokens[0])
                items = [int(x) for x in tokens[1:]]
                if len(items) == 0:
                    continue
                # last two are test and valid (following VA-GNN convention)
                if len(items) >= 2:
                    test_dict[uid]  = items[-1]
                    val_dict[uid]   = items[-2]
                    train_items_u   = items[:-2]
                elif len(items) == 1:
                    test_dict[uid]  = items[0]
                    train_items_u   = []
                else:
                    train_items_u   = items

                for it in train_items_u:
                    train_users.append(uid)
                    train_items.append(it)
                allPos[uid] = train_items_u

        return (np.array(train_users, dtype=np.int64),
                np.array(train_items, dtype=np.int64),
                val_dict, test_dict, allPos)

    @staticmethod
    def _load_allpos_only(filepath, n_users):
        """Load a file into user -> [positive ids] without train/val/test split."""
        allPos = [[] for _ in range(n_users)]
        if not os.path.exists(filepath):
            return allPos
        with open(filepath) as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                uid   = int(tokens[0])
                items = [int(x) for x in tokens[1:]]
                allPos[uid] = items
        return allPos

    @staticmethod
    def _build_bipartite_sparse(users, items, n_users, n_items):
        """
        Build a D^{-1/2} A D^{-1/2} normalised sparse adjacency for LightGCN-style propagation.
        Shape: (n_users + n_items) x (n_users + n_items)
        Returned as a torch.sparse_coo_tensor on config.DEVICE.
        """
        n = n_users + n_items

        # Build R (user x item) scipy sparse
        data   = np.ones(len(users), dtype=np.float32)
        R      = sp.csr_matrix((data, (users, items)), shape=(n_users, n_items))

        # Adjacency block: A = [[0, R], [R^T, 0]]
        adj    = sp.block_array([[None, R], [R.T, None]]).tocoo().astype(np.float32)

        # Degree normalisation: D^{-1/2}
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv  = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        D_inv  = sp.diags(d_inv)
        norm_adj = D_inv @ adj @ D_inv
        norm_adj = norm_adj.tocoo()

        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values  = torch.FloatTensor(norm_adj.data)
        sparse  = torch.sparse_coo_tensor(indices, values, (n, n)).to(config.DEVICE)
        if config.DEVICE.type == "cpu":
            try:
                sparse = sparse.to_sparse_csr()
            except Exception:
                pass
        return sparse

    # ── PyTorch Dataset interface (not used for training; BPR sampler used instead) ──
    def __len__(self):
        return self.trainDataSize

    def __getitem__(self, idx):
        return self._train_uv_users[idx], self._train_uv_videos[idx]
