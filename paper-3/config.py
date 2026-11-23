"""
config.py – Hyperparameter configuration for Paper 3 (MPGCL) Benchmark on MovieLens-1M.
"""

import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "other-dataset's", "movielens", "ml-1m"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hardware Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture (Paper 3 Setup)
EMBED_DIM = 200           # Fixed to d=200 matching Paper 3 Section 5.1.3
GNN_LAYERS = 3            # 3-hop bipartite graph aggregation
PROJECTION_DIM = 200      # Projection head dimension for contrastive learning
DROPOUT = 0.1
TEMPERATURE = 0.2         # NT-Xent temperature parameter tau (Equation 8)

# Meta-Path Homogeneous Graph Settings (Idea C)
MAX_WALK_LEN = 7          # Max walk length for meta-path sampling (Section 5.1.3)
META_TOP_K = 10           # Top-K neighbors retained in homogeneous co-occurrence graphs
PERTURB_RATIO = 0.1       # Graph augmentation edge perturbation ratio eta (Equation 6)

# Graph Denoising Settings (Vynora Edge)
DENOISE_INTERVAL = 5      # Apply edge pruning every D=5 epochs
DENOISE_DROP_RATE = 0.05  # Filter bottom 5% unreliable noisy interaction edges
DENOISE_EMA_BETA = 0.7    # Exponential moving average smoothing factor

# Loss Weights (Equation 14: L = beta * L_un + gamma * L_su)
BETA_CONTRASTIVE = 0.1    # Contrastive regularizer weight (standard SimGCL/MPGCL scale)
GAMMA_SUPERVISED = 1.0    # Supervised BPR ranking loss weight (primary objective)
WEIGHT_DECAY = 1e-4       # L2 regularizer lambda

# Training Budget
EPOCHS = 40
BATCH_SIZE = 4096         # Mini-batch BPR negative sampling
LR = 0.001
SEED = 42

# Evaluation & Diversity Settings
TOP_K = 10                # Primary benchmark position K=10 (Paper 3 Table 5)
LAMBDA_DIV = 0.0          # 0.0 = pure accuracy (compare with Paper 3); >0.0 = genre diversity
