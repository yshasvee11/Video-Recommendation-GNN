# Vynora — Creator-Aware Micro-Video Recommendation with MMR Diversification

A research implementation of a **creator-aware heterogeneous graph recommender** for micro-video platforms, with an MMR-based post-hoc diversity re-ranking stage.

> [!IMPORTANT]
> **Honest framing note:** This codebase was developed by studying five foundational papers (listed below). It is *not* a faithful reproduction of all five. Each section below documents exactly what was implemented versus what the original paper requires. Claims are scoped to what is actually working in the code.

---

## What This System Does

Given a user's past micro-video interactions and vlogger (creator) subscriptions, Vynora:

1. Learns separate user, video, and vlogger embeddings via dual-view GNN propagation over a tripartite interaction graph (User–Video and User–Vlogger adjacency matrices).
2. Scores candidate videos as a weighted combination of video-view and vlogger-view dot products.
3. Optionally applies **MMR re-ranking** (Maximal Marginal Relevance) to diversify the final top-K list by creator, preventing all recommendations coming from the same vlogger.

The core novelty is the **creator-aware scoring** (VA-GNN inspired): the final relevance score for `(user, video)` accounts for both the video content and the affinity toward the video's creator.

---

## Reference Papers

| Paper | Reference | What We Actually Use |
|---|---|---|
| **VA-GNN** | [Paper 4](research-papers/Paper_4_VA-GNN.pdf) | Tripartite graph structure (User, Video, Vlogger), dual-view propagation, lambda-weighted scoring. **Note:** VA-GNN uses a learned adaptive gate; we use a fixed lambda=0.5. Full creator-video propagation during message passing is not yet implemented. |
| **MPGCL** | [Paper 3](research-papers/Paper_3-MPGCL.pdf) | Cross-view InfoNCE contrastive loss, ProjectionHead (GELU + residual + LayerNorm). **Note:** MPGCL builds meta-path-derived homogeneous graphs; we contrast the two direct interaction views (User-Video vs User-Vlogger). This is a simpler dual-view contrast, not MPGCL meta-path graphs. |
| **CONDE** | [Paper 2](research-papers/Paper_2-CONDE.pdf) | Iterative edge pruning with EMA-accumulated reliability scores, logarithmic drop schedule. **Note:** CONDE uses a TransformerEncoder over item sequences and concept-aware personalized denoising. We use a lightweight MLP edge scorer — structurally similar, but not concept-aware and not trained end-to-end through the recommendation loss. |
| **Paper 5** | [Paper 5 (Chun et al., ESWA 2024)](research-papers/Paper_5.pdf) | Publisher Diversity metric (Eq 1, Gini-Simpson), Content Diversity metric (Eq 1-2), MMR re-ranking baseline. **Note:** Paper 5's primary contribution is a GNN that *predicts* diversification scores and uses those predictions to re-rank. The DiversificationGNN class (`diversity.py`) implements this architecture, but the trained checkpoint is NOT currently loaded at inference — `score_rerank` computes empirical labels at run time, making it an empirical re-ranking baseline, not Paper 5's predicted-score approach. |
| **Sagittarius** | [Paper 1](research-papers/Paper%201-Sagittarius.pdf) | **Not implemented.** Sagittarius requires behavior-type annotations (click, watch, share, like) and timestamps; the dataset has neither. |

---

## Project Structure

```
Video-Recommendation-GNN/
├── data/                              ← Benchmark datasets (takatak & wechat)
│   ├── takatak/                       ← 14,571 users, 13,133 videos, 912 vloggers
│   └── wechat/                        ← 19,737 users, 25,976 videos, 2,088 vloggers
├── research-papers/                   ← The 5 foundational reference papers (PDFs)
│   ├── Paper 1-Sagittarius.pdf
│   ├── Paper_2-CONDE.pdf
│   ├── Paper_3-MPGCL.pdf
│   ├── Paper_4_VA-GNN.pdf
│   └── Paper_5.pdf
├── results/                           ← Benchmark logs, ablation and training results
│   ├── results_takatak.json           ← Takatak training & validation logs
│   ├── results_wechat.json            ← WeChat training & validation logs
│   └── ablation_summary.json          ← 6-variant ablation (tests Vynora recommender, not Paper 5)
├── config.py                          ← Hyperparameters, paths & diversity settings
├── dataloader.py                      ← VynoraDataset: loads takatak/wechat, builds sparse graphs
├── model.py                           ← VynoraGNN: dual-view GNN + contrastive + edge scorer
├── sampler.py                         ← BPR dual-stream negative sampler (VA-GNN pattern)
├── denoiser.py                        ← Edge reliability-based graph pruning (CONDE-inspired)
├── diversity.py                       ← Diversity metrics, MMR re-ranking, experimental DiversificationGNN
├── train_diversification_predictor.py ← [EXPERIMENTAL] Trains Paper 5 GNN (not yet used at inference)
├── diversity_benchmark.py             ← Accuracy vs. Diversity Pareto trade-off evaluator
├── evaluate.py                        ← Recall@K, NDCG@K, Precision@K, MRR@K + Diversity@K
├── train.py                           ← Full training loop (run this first)
├── predict.py                         ← Single-user inference with optional MMR diversity
├── ablation.py                        ← Six-variant ablation of the Vynora recommender
├── generate_colab_nb.py               ← Colab notebook builder
├── vynora2_colab.ipynb                ← Interactive Google Colab T4 GPU execution notebook
├── vynora_single_file_train.ipynb     ← Standalone all-in-one Colab training & diversity notebook
└── requirements.txt                   ← Dependencies specification
```

---

## Google Colab Execution

You can train the model, evaluate recommendations, and benchmark the accuracy-diversity trade-off on a free **Google Colab T4 GPU** with 1 click:

| Notebook | Description | Open in Colab |
|---|---|---|
| **Interactive Workflow** (`vynora2_colab.ipynb`) | Form sliders for hyperparameters, live loss/recall plots, Pareto trade-off curves, and creator distribution charts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yshasvee11/Video-Recommendation-GNN/blob/main/vynora2_colab.ipynb) |
| **Standalone All-in-One** (`vynora_single_file_train.ipynb`) | 100% self-contained single `.ipynb` file with model, dataset, MMR diversification, training loop, and metrics embedded directly in cells | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yshasvee11/Video-Recommendation-GNN/blob/main/vynora_single_file_train.ipynb) |

---

## Data

Uses the VA-GNN dataset format:

```
data_size.txt            n_users  n_videos  n_vloggers
user_video_data.txt      user_id  v1 v2 ... vN  (last 2 = val, test items per VA-GNN protocol)
user_vlogger_data.txt    user_id  a1 a2 ... aN
user_vlogger_data2.txt   user_id  a1 a2 ... aN  (auxiliary vlogger stream)
vlogger_video_data.txt   vlogger_id  video_id   (tab-separated creator-video pairs)
```

| Dataset   | Users  | Videos | Vloggers |
|-----------|--------|--------|----------|
| takatak   | 14,571 | 13,133 |    912   |
| wechat    | 19,737 | 25,976 |  2,088   |

> [!NOTE]
> The dataset does not contain timestamps or behavior-type annotations (click/watch/like/share). Interaction order within each user's row is **not confirmed chronological**. The train/val/test split follows the VA-GNN protocol: the last item in a user's row is the test item, the second-to-last is the validation item, and the rest are training interactions. This is a standard leave-one-out split for implicit feedback.

---

## Setup

```bash
pip install torch scipy numpy tqdm
```

No PyTorch Geometric or DGL required — all graph operations use native PyTorch sparse tensors.

---

## Training

```bash
# Train on TakaTak (default)
python train.py --dataset takatak

# Train on WeChat
python train.py --dataset wechat
```

---

## Inference & MMR Diversity Re-Ranking

Run single-user inference with optional MMR creator-diversity re-ranking:

```bash
# Pure relevance baseline (no diversity)
python predict.py --user 42 --topk 10 --dataset takatak

# MMR re-ranking with lambda=0.4 (balance: relevance + publisher diversity)
python predict.py --user 42 --topk 10 --dataset takatak --diversity 0.4

# Sweep: lambda=0.0 to lambda=1.0 across the full validation set
python diversity_benchmark.py --dataset takatak --mode val
```

### Example Output (User 42, lambda=0.4, MMR):

```
========================================================
  Home Screen: Top-10 for User 42  (takatak)
  Mode: Diversity (lambda=0.4, publisher, mmr)
========================================================
Rank  Video ID   Vlogger     Score
------------------------------------
   1       284        64    6.0992
   2        43        36    5.3996
   3         2         2    5.3651
   ...
  10         6         6    4.8263
------------------------------------

Diversity Analysis (Top-10):
  - Publisher Diversity (Eq 1):  1.0000
  - Unique Vloggers:             10 / 10

Diversity Improvement vs Baseline:
  - Publisher Diversity Gain:    +28.6%  (0.7778 -> 1.0000)
  - Creator Variety Increase:    +4 unique vloggers (6 -> 10)
```

---

## Accuracy vs. Diversity Trade-off

Run the full Pareto evaluation sweeping lambda from 0.0 to 1.0:

```bash
python diversity_benchmark.py --dataset takatak --mode val
```

### Experimental Results on Takatak (Validation Set):

| lambda_div | Recall@10 | NDCG@10 | Publisher Diversity@10 | Catalog Coverage@10 |
|---|---|---|---|---|
| **0.00** (Base) | 0.0436 | 0.0216 | 0.8424 | 0.0222 |
| **0.30** (MMR)  | **0.0472** | **0.0220** | **0.9928** | **0.0228** |

> [!NOTE]
> Publisher Diversity at lambda=0.3 improves from **0.8424 to 0.9928** (+17.8%). The modest Recall improvement (0.0436 → 0.0472) is consistent with MMR recovering items the relevance-only model missed due to vlogger redundancy. Content Diversity values are low (≈0.013) because the content distance is computed over collaborative latent embeddings (learned from interaction co-occurrence), not textual or semantic video features — this is a limitation acknowledged in future work.

---

## Ablation Study

Six variants of the Vynora **recommender** (not the Paper 5 predictor):

| Variant | L_rank | Vlogger view | Cross-view CL | Edge Pruning |
|---------|--------|---------|-----|-----------
| A. Baseline | ✓ | – | – | – |
| B. + Vlogger | ✓ | ✓ | – | – |
| C. + CL | ✓ | ✓ | ✓ | – |
| D. + Denoising | ✓ | – | – | ✓ |
| E. Vlogger + Denoise | ✓ | ✓ | – | ✓ |
| F. Full model | ✓ | ✓ | ✓ | ✓ |

```bash
python ablation.py
```

> [!WARNING]
> Current ablation results show the Full model (F) at Recall@10=0.0468, below the simple Baseline (A) at 0.0484 and the Vlogger-only variant (B) at 0.0471. This means the contrastive + edge-pruning combination is not yet empirically justified with the current hyperparameters. The ablation is retained for transparency.

---

## Limitations & Future Work

| Limitation | Description |
|---|---|
| **No learned gate** | The video/vlogger score mixture uses a fixed lambda=0.5. VA-GNN learns an adaptive gate conditioned on the candidate video and its creator. |
| **Simpler contrastive views** | Cross-view CL contrasts User-Video vs User-Vlogger interaction views. MPGCL uses separate meta-path-derived homogeneous graphs (e.g., User-Video-User, User-Vlogger-User meta-paths). |
| **MLP edge scorer** | The edge reliability scorer is a 2-layer MLP on embedding pairs. CONDE's scorer uses a TransformerEncoder over a user's full item-visit sequence, with personalized concept selection. |
| **No content features** | Content Diversity is computed over collaborative filtering embeddings, not TF-IDF or video tags. Paper 5 uses TF-IDF of video text. |
| **No genre diversity** | Paper 5 defines separate genre, publisher, and content diversification targets. Genre metadata is unavailable in this dataset. |
| **Predictor disconnected** | The DiversificationGNN in `diversity.py` is trained by `train_diversification_predictor.py` (Val MAE≈0.44, weak — targets are tightly clustered near 0.97). The saved checkpoint is not loaded during inference. The `score_rerank` method recomputes empirical labels at runtime. This is a valid empirical re-ranking baseline but is **not** Paper 5's predicted-score re-ranking. |
| **No chronological split** | Interaction timestamps are unavailable; proper chronological train/val/test splits cannot be established. |
| **Single seed** | All results are from a single training run. Paper-quality results require mean ± std across 5+ seeds. |

---

## Citation

If you use this codebase, please cite the foundational papers this work draws from:

- **VA-GNN**: Vlogger-Aware Graph Neural Network for Micro-Video Recommendation
- **MPGCL**: Meta-Path-Guided Graph Contrastive Learning (MPGCL)
- **CONDE**: CONDE — Concept-Aware Denoising for Graph-based Social Recommendation
- **Chun et al. (2024)**: "Predicting diversification scores of videos in recommendation network", Expert Systems With Applications 238 (2024) 121803
