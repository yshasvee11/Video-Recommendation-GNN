"""
config.py  –  Vynora 2.0: Denoising & Vlogger-Aware Meta-Path Graph CL
Central configuration: all hyper-parameters and path settings live here.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_data_root():
    # 1. Environment variable override
    env_dir = os.environ.get("VYNORA_DATA_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir
    # 2. Local repository data directory
    p_data = os.path.abspath(os.path.join(BASE_DIR, "data"))
    if os.path.exists(p_data):
        return p_data
    p_local_vagnn = os.path.abspath(os.path.join(BASE_DIR, "VAgnn-dataset"))
    if os.path.exists(p_local_vagnn):
        return p_local_vagnn
    # 3. Local repo structure (relative to workspace)
    p_vagnn = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "VAGNN", "data"))
    if os.path.exists(p_vagnn):
        return p_vagnn
    p_other = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "other-dataset's"))
    if os.path.exists(p_other):
        return p_other
    p1 = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "VAgnn-dataset"))
    if os.path.exists(p1):
        return p1
    # 4. Colab direct paths: /content/VAgnn-dataset or ./VAgnn-dataset
    p2 = "/content/VAgnn-dataset"
    if os.path.exists(p2):
        return p2
    p3 = os.path.abspath(os.path.join(BASE_DIR, "VAgnn-dataset"))
    if os.path.exists(p3):
        return p3
    p4 = os.path.abspath(os.path.join(BASE_DIR, "..", "VAgnn-dataset"))
    if os.path.exists(p4):
        return p4
    return p_vagnn

DATA_ROOT  = _get_data_root()
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
DATASET = "wechat"             # "takatak" | "wechat"

# ─────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────
EMBED_DIM        = 64          # latent dimension for all nodes
N_LAYERS         = 3           # number of graph propagation layers
PROJECTION_DIM   = 64          # contrastive projection head output dim
PROJECTION_DROP  = 0.1         # dropout inside projection head
DENOISE_TOPK     = 10          # top-k neighbors to KEEP during denoising (0 = disabled)
LAMBDA_VLOGGER   = 0.5         # lambda for vlogger-view score blend: s = lambda*s_vid + (1-lambda)*s_vlog

# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
BPR_BATCH       = 16384 if DEVICE.type == "cpu" else 4096
NEG_K           = 1            # number of negatives per positive
EPOCHS          = 200
PATIENCE        = 10           # early-stop patience (epochs without val improvement)
SEED            = 42

# ─────────────────────────────────────────────────────────────
# Loss coefficients   L = L_rank + beta*L_cl + gamma*L_den + delta*L_vlog
# ─────────────────────────────────────────────────────────────
CL_REG          = 0.1          # beta  - cross-view contrastive loss weight
DENOISE_REG     = 0.05         # gamma - denoising consistency loss weight
VLOGGER_REG     = 0.1          # delta - auxiliary user-vlogger BPR loss weight
TEMPERATURE     = 0.2          # InfoNCE temperature

# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
TOPK_LIST       = [10, 20]     # Recall@K and NDCG@K
TEST_BATCH      = 100          # users per evaluation batch

# ─────────────────────────────────────────────────────────────
# Diversity Recommendation (Paper 5: Chun et al., ESWA 2024)
# ─────────────────────────────────────────────────────────────
DIVERSITY_LAMBDA = 0.3          # 0.0 = pure relevance, 1.0 = pure diversity
DIVERSITY_TYPE   = "publisher"  # "publisher" | "content" | "hybrid"
DIVERSITY_METHOD = "mmr"        # "mmr" (Paper 5 Sec 2.1) | "score_rerank" (Paper 5 Sec 7)
CANDIDATE_POOL   = 100          # Top-M candidate pool size before diverse re-ranking

