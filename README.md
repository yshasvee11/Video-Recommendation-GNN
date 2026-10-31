# Vynora — Creator-Aware Micro-Video Recommendation & Fair Diversification

Vynora is a high-performance PyTorch implementation of creator-aware micro-video recommendation. It models the bipartite graph of users, micro-videos, and creators (vloggers) with dual-view message passing, adaptive bilinear gating, and cross-view contrastive alignment, followed by fair re-ranking for publisher diversity.

---

## Benchmark Results

### 1. TakaTak Dataset
- **Relevance**: Recall@10: `0.0651` | NDCG@10: `0.0382` | Recall@20: `0.1009` | NDCG@20: `0.0471`
- **Creator Diversification**: MMR (λ=0.3) maintains high relevance while providing balanced creator distribution.

### 2. WeChat Dataset (Uncompromised Full-Epoch Training)
- **Model Checkpoints**: `best_wechat_vynora_max.pt` (best validation weights) & `last_wechat_vynora_model.pt`
- **Leave-One-Out Test Performance**:
  - Pure Relevance: Recall@10: `0.0034` | NDCG@10: `0.0023` | Pub Diversity: `0.0061`
  - **Hard Creator Cap (Max 2)**: **Recall@10: `0.0073` (+114.7%)** | NDCG@10: `0.0038` | **Pub Diversity: `0.9052` (+14,700%)**
  - **MMR Publisher (λ=0.5)**: Recall@10: `0.0064` (+88.2%) | NDCG@10: `0.0037` | Pub Diversity: `0.5819`

> **Key Finding**: In creator-dense micro-video feeds, unconstrained ranking causes severe creator starvation and cannibalization. Constraining recommendations to at most 2 videos per creator doubles Recall@10 and eliminates winner-take-all creator concentration.

---

## Consolidated Research Foundations

All 7 core research papers are consolidated into `research-papers/`:

| Paper ID | File | Reference | Core Contribution to Vynora |
|---|---|---|---|
| **Paper 1** | [`Paper_1_Sagittarius.pdf`](file:///research-papers/Paper_1_Sagittarius.pdf) | Zhuo et al. | Interaction log-frequency weighting ($w_{uv} = \log(1 + c_{uv})$) & weighted BPR. |
| **Paper 2** | [`Paper_2_CONDE.pdf`](file:///research-papers/Paper_2_CONDE.pdf) | CIKM 2021 | Concept-aware soft graph filtering and noise attenuation. |
| **Paper 3** | [`Paper_3_MPGCL.pdf`](file:///research-papers/Paper_3_MPGCL.pdf) | He et al., ESWA 2023 | Meta-path graph contrastive learning across homogeneous subgraphs. |
| **Paper 4** | [`Paper_4_VA-GNN.pdf`](file:///research-papers/Paper_4_VA-GNN.pdf) | Lai et al. | Vlogger-augmented dual GCN with bilinear adaptive gating ($w = \sigma(e_v^T Q e_{p_v})$). |
| **Paper 5** | [`Paper_5_Diversity_Prediction.pdf`](file:///research-papers/Paper_5_Diversity_Prediction.pdf) | Chun et al., ESWA 2024 | Creator diversification scoring and fair candidate re-ranking. |
| **Paper 6** | [`Paper_6_Multimodal_Analysis.pdf`](file:///research-papers/Paper_6_Multimodal_Analysis.pdf) | Zhou et al., NTU (Aug 2025, arXiv:2508.05377) | Critical analysis of multimodality in RecSys, late-fusion ensembling, cold-start mitigation. |
| **Paper 7** | [`Paper_7_Frozen_LVLM_DFF.pdf`](file:///research-papers/Paper_7_Frozen_LVLM_DFF.pdf) | Sun et al., ICMR (May 2026, arXiv:2512.21863v2) | Frozen LVLM visual-language feature extraction & Disentangled Feature Fusion (DFF). |

---

## Quick Start

```powershell
pip install -r requirements.txt

# Train on TakaTak
python train.py --dataset takatak

# Train on WeChat
python train.py --dataset wechat

# Compare re-ranking strategies on candidate pool
python diversity_benchmark.py --dataset takatak --mode val --compare

# Run research ablations
python ablation.py
```

---

## Directory Layout

```
data/                                Bundled TakaTak and WeChat interaction data
research-papers/                     7 consolidated and standardized research papers
results/                             Standardized trained models, notebooks, and JSON benchmarks
  ├── best_takatak_model.pt          Best trained TakaTak model weights
  ├── best_wechat_vynora_max.pt      Best WeChat validation checkpoint
  ├── last_wechat_vynora_model.pt    Final WeChat training checkpoint
  ├── vynora_wechat_max_performance.ipynb Executed Colab training notebook with full logs
  ├── vynora_single_file_train_takatak.ipynb Executed TakaTak benchmark notebook
  ├── results_wechat_max.json        WeChat benchmark metrics & fair re-ranking comparison
  └── results_takatak.json           TakaTak benchmark metrics
config.py                            Hyperparameter configuration
dataloader.py                        Data loading, graph construction, leave-one-out splits
model.py                             Creator-aware dual-view GNN with bilinear gating
sampler.py                           BPR uniform negative sampling
denoiser.py                          Graph-edge soft filtering
evaluate.py                          Ranking (Recall, NDCG, MRR) and diversity metrics
diversity.py                         MMR and creator diversification re-ranking
ablation.py                          6-variant research ablation runner
```

---

## Data Format

Each dataset directory (`data/takatak/` and `data/wechat/`) contains:
- `data_size.txt`: Entity counts (users, videos, creators)
- `user_video_data.txt`: User-to-video implicit interactions
- `user_vlogger_data.txt`: Primary user-to-creator interactions
- `user_vlogger_data2.txt`: Auxiliary user-to-creator interactions
- `vlogger_video_data.txt`: Creator-to-video publishing associations

Evaluation follows strict leave-one-out protocol: user's final interaction is held out for test, the preceding one is for validation, and earlier interactions form the training graph.
