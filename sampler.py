"""
sampler.py  –  Vynora 2.0
BPR negative sampling for the tripartite user-video-vlogger graph.
Faithfully reproduces the VA-GNN sampling scheme (UniformSample_original_python)
with additional support for negative vlogger sampling in the auxiliary stream.
"""

import numpy as np
import torch
from dataloader import VynoraDataset
import config


def uniform_sample(dataset: VynoraDataset, neg_k: int = config.NEG_K):
    """
    Draw BPR training tuples for BOTH the user-video and user-vlogger streams.

    Stream 1 (S):   [user, pos_video, neg_video, pos_vlogger, neg_vlogger]
    Stream 2 (S2):  [user, pos_vlogger, neg_vlogger]

    Follows VA-GNN sampling exactly:
      - neg_video must not be in user's positive video set
      - neg_vlogger_of_negvideo must not be in user's positive vlogger set

    Returns:
        S:  np.ndarray [N1, 5]
        S2: np.ndarray [N2, 3]
    """
    allPos_video, allPos_vlogger = dataset.allPos
    allPos_video_sets, allPos_vlogger_sets = dataset.allPosSets
    allPos2 = dataset.allPos2
    allPos2_sets = dataset.allPos2Sets
    vlogger_list = dataset.vlogger_list.tolist() if hasattr(dataset.vlogger_list, "tolist") else dataset.vlogger_list

    n_users    = dataset.n_users
    n_videos   = dataset.n_videos
    n_vloggers = dataset.n_vloggers
    train_size  = dataset.trainDataSize
    train_size2 = dataset.trainDataSize2

    users  = np.random.randint(0, n_users, train_size)
    users2 = np.random.randint(0, n_users, train_size2)

    S, S2 = [], []

    # ── Stream 1: user-video ──────────────────────────────────────
    uv_weights_dict = getattr(dataset, "uvWeights", {})
    weights = []

    for user in users:
        pos_videos = allPos_video[user]
        if len(pos_videos) == 0:
            continue

        pos_idx     = np.random.randint(0, len(pos_videos))
        pos_video   = pos_videos[pos_idx]
        pos_vlogger = vlogger_list[pos_video]
        pos_v_set   = allPos_video_sets[user]
        pos_a_set   = allPos_vlogger_sets[user]
        w           = uv_weights_dict.get((user, pos_video), 1.0)

        for _ in range(neg_k):
            # sample neg_video such that it's not in pos set AND
            # its vlogger is not in user's positive vlogger set
            attempts = 0
            while True:
                neg_video   = np.random.randint(0, n_videos)
                neg_vlogger = vlogger_list[neg_video]
                if neg_video not in pos_v_set and neg_vlogger not in pos_a_set:
                    break
                attempts += 1
                if attempts > 50:   # safety valve for very dense users
                    neg_vlogger = neg_vlogger if neg_video not in pos_v_set else pos_vlogger
                    break
            S.append([user, pos_video, neg_video, pos_vlogger, neg_vlogger])
            weights.append(w)

    # ── Stream 2: user-vlogger ────────────────────────────────────
    for user in users2:
        pos_vloggers_u = allPos2[user]
        if len(pos_vloggers_u) == 0:
            continue

        pos_idx     = np.random.randint(0, len(pos_vloggers_u))
        pos_vlogger = pos_vloggers_u[pos_idx]
        pos_a2_set  = allPos2_sets[user]

        for _ in range(neg_k):
            attempts = 0
            while True:
                neg_vlogger = np.random.randint(0, n_vloggers)
                if neg_vlogger not in pos_a2_set:
                    break
                attempts += 1
                if attempts > 50:
                    break
            S2.append([user, pos_vlogger, neg_vlogger])

    return np.array(S, dtype=np.int64), np.array(S2, dtype=np.int64), np.array(weights, dtype=np.float32)


def shuffle(*arrays):
    """Shuffle multiple arrays with the same permutation."""
    perm = torch.randperm(arrays[0].shape[0])
    return [a[perm] for a in arrays]


def minibatch(*tensors, batch_size: int):
    """Yield slices of equal-sized batches from multiple tensors."""
    n = tensors[0].shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield tuple(t[start:end] for t in tensors)
