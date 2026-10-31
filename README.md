# Vynora — Creator-Aware Micro-Video Recommendation with Diversification

A research-grounded micro-video recommendation system synthesizing methodologies from five foundational papers, adapted to the empirical constraints of implicit-feedback micro-video platforms (TakaTak and WeChat).

> [!IMPORTANT]
> **Honest Framing & Research Grounding:** This codebase implements and adapts key concepts from five reference papers (listed below). Because our datasets provide implicit user interactions and creator-video mappings without timestamps, behavior types (click/share/like), or video text/tags, each paper's techniques are implemented strictly within what the data genuinely supports. All mathematical adaptations are documented explicitly below.

---

## Key System Features

1. **Faithful VA-GNN Bilinear Adaptive Gate (Eq. 17):**
   Candidate video scoring dynamically balances video content affinity and creator affinity via a learned bilinear interaction matrix $Q \in \mathbb{R}^{d \times d}$:
   $$w = \sigma(e_v^T Q e_{p_v}), \quad \hat{y}_{uvi} = w \cdot (e_u^T e_v) + (1 - w) \cdot (e_u^T e_{p_v})$$
2. **3-Entity Cross-View Contrastive Learning:**
   InfoNCE contrastive alignment across User-Video and User-Vlogger views spans all three entity types:
   $$\mathcal{L}_{CL} = \frac{1}{3}\left(\mathcal{L}_u + \mathcal{L}_v + \mathcal{L}_p\right)$$
3. **Sagittarius Frequency-Weighted Graph & BPR:**
   Edge weights reflect repeated interactions ($w_{uv} = \log(1 + c_{uv})$), driving a frequency-weighted BPR ranking objective:
   $$\mathcal{L}_{rank} = -\sum_{(u, v_i, v_j) \in \mathcal{D}} w_{u, v_i} \ln \sigma(\hat{y}_{u, v_i} - \hat{y}_{u, v_j})$$
4. **MPGCL Homogeneous Meta-Path Graphs:**
   Sparse User–Video–User ($A_U = RR^T$) and Video–User–Video ($A_V = R^TR$) co-occurrence graphs with top-$k$ pruning and disk caching (`.npz`) for meta-path contrastive regularization.
5. **CONDE Soft Interaction Filtering:**
   Differentiable edge reliability filtering ($r_{uv} = \text{MLP}([e_u \Vert e_v \Vert e_u \odot e_v])$) to downweight spurious or noisy implicit interactions.
6. **Calibrated Paper 5 Diversification & Fair Re-Ranking:**
   Deduplicated historical interaction sets for empirical publisher diversification ($S_{v,pub}$), rigorous separation between Publisher Diversity and **Embedding-Space Diversity**, and a controlled 6-way fair re-ranking comparison evaluated from an identical Top-100 candidate pool.

---

## Reference Papers & Implementation Fidelity

| Paper | Reference | Implementation & Dataset Adaptation |
|---|---|---|
| **VA-GNN** | [Paper 4](research-papers/Paper_4_VA-GNN.pdf) | **Faithful Core Implementation.** Tripartite graph (User, Video, Vlogger), dual-view GNN message passing, auxiliary vlogger BPR stream, exact **Bilinear Adaptive Gate** ($w = \sigma(e_v^T Q e_{p_v})$, Eq. 17), and 3-entity cross-view contrastive loss ($\mathcal{L}_u + \mathcal{L}_v + \mathcal{L}_p$). |
| **Sagittarius** | [Paper 1](research-papers/Paper%201-Sagittarius.pdf) | **Interaction-Frequency Adaptation.** Because the datasets lack behavior types (click/watch/share/like) and timestamps, we adapt Sagittarius by tracking interaction repetition count $c_{uv}$ with logarithmic scaling ($w_{uv} = \log(1 + c_{uv})$) and frequency-weighted BPR. |
| **MPGCL** | [Paper 3](research-papers/Paper_3-MPGCL.pdf) | **Sparse Meta-Path Adaptation.** Implements User-Video-User ($RR^T$) and Video-User-Video ($R^TR$) homogeneous meta-path graphs with chunked batch dot products, top-$k$ pruning ($k=15$), and `.npz` caching to prevent dense $O(N^2)$ memory explosion. |
| **CONDE** | [Paper 2](research-papers/Paper_2-CONDE.pdf) | **Soft Relevance Denoising.** Because concept annotations (tags/hashtags) are absent, we implement differentiable edge reliability scoring ($r_{uv} = \text{MLP}([e_u \Vert e_v \Vert e_u \odot e_v])$) and iterative reliability-based graph pruning. |
| **Paper 5** | [Chun et al. (ESWA 2024)](research-papers/Paper_5.pdf) | **Calibrated Diversity & Fair Re-Ranking.** Implements publisher diversification (Gini-Simpson ordered pairs) with deduplicated user histories, embedding-space diversity, and controlled fair candidate-pool re-ranking comparing Pure Relevance, Hard Creator Cap, MMR Publisher Div, MMR Embedding Div, Hybrid MMR, and Score Blending. |

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
│   ├── fair_reranking_comparison_takatak.json ← Controlled 6-way re-ranking benchmark
│   └── ablation_summary.json          ← Full ablation study logs
├── config.py                          ← Hyperparameters, paths & research flags
├── dataloader.py                      ← VynoraDataset: graphs, frequency counts, meta-paths
├── model.py                           ← VynoraGNN: bilinear gate, 3-entity CL, soft filter
├── sampler.py                         ← BPR sampler with Sagittarius frequency weights
├── denoiser.py                        ← Edge reliability-based graph pruning (CONDE)
├── diversity.py                       ← Diversity metrics, MMR re-ranking, DiversificationGNN
├── diversity_benchmark.py             ← Fair candidate-pool re-ranking & Pareto evaluator
├── evaluate.py                        ← Evaluation metrics (Recall, NDCG, Diversity)
├── train.py                           ← Full training loop with early stopping
├── predict.py                         ← Single-user inference with adaptive gating & MMR
├── ablation.py                        ← 8-variant research ablation matrix (M1–M8)
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
| **Interactive Workflow** (`vynora2_colab.ipynb`) | Form sliders for hyperparameters, live loss/recall plots, Pareto trade-off curves, fair re-ranking comparison, and creator distribution charts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yshasvee11/Video-Recommendation-GNN/blob/main/vynora2_colab.ipynb) |
| **Standalone All-in-One** (`vynora_single_file_train.ipynb`) | 100% self-contained single `.ipynb` file with model, dataset, Sagittarius weighting, bilinear gate, MMR diversification, training loop, and metrics embedded directly in cells | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yshasvee11/Video-Recommendation-GNN/blob/main/vynora_single_file_train.ipynb) |

---

## Dataset Characteristics

The data follows the implicit-feedback tripartite structure:

```
data_size.txt            n_users  n_videos  n_vloggers
user_video_data.txt      user_id  v1 v2 ... vN  (leave-one-out: last=test, 2nd-last=val)
user_vlogger_data.txt    user_id  a1 a2 ... aN  (primary vlogger stream)
user_vlogger_data2.txt   user_id  a1 a2 ... aN  (auxiliary vlogger stream)
vlogger_video_data.txt   vlogger_id  video_id   (tab-separated creator-video mapping)
```

| Dataset | Users | Videos | Vloggers | Sparsity |
|---|---|---|---|---|
| **TakaTak** | 14,571 | 13,133 | 912 | 99.88% |
| **WeChat** | 19,737 | 25,976 | 2,088 | 99.93% |

> [!NOTE]
> The datasets provide implicit interaction lists without timestamps. Per standard literature protocol (VA-GNN, LightGCN), evaluation uses a leave-one-out split: the last item in each user row is held out for testing, the second-to-last for validation, and earlier items form the training graph.

---

## Training & Evaluation

```bash
# Train on TakaTak with Bilinear Adaptive Gate & 3-Entity CL
python train.py --dataset takatak

# Train on WeChat
python train.py --dataset wechat

# Evaluate single-user recommendations with MMR diversity
python predict.py --user 42 --topk 10 --dataset takatak --diversity 0.4
```

---

## Controlled Fair Re-Ranking Benchmark

To rigorously test whether diversity hurts or helps recommendation performance, all re-ranking strategies are evaluated from the **exact same candidate pool** (Top-100 items scored by the base recommender) across all 14,571 validation users:

```bash
python diversity_benchmark.py --dataset takatak --mode val --compare
```

### Empirical Results on TakaTak (Full Validation Set, 14,571 Users):

| Re-Ranking Method | Recall@10 | NDCG@10 | Publisher Diversity@10 | Embedding-Space Diversity@10 | Catalog Coverage@10 |
|---|---|---|---|---|---|
| **1. Pure Relevance** (Base) | 0.0154 | 0.0081 | 0.0239 | 0.0965 | 0.0251 |
| **2. Hard Creator Cap** (max 2/creator) | 0.0154 | 0.0081 | 0.0239 | 0.0965 | 0.0251 |
| **3. MMR Publisher Div** ($\lambda=0.3$) | **0.0171** (+11.0%) | **0.0089** (+9.8%) | **0.1846** (+672.6%) | 0.0914 | 0.0241 |
| **4. MMR Embedding Div** ($\lambda=0.3$) | 0.0134 (-13.0%) | 0.0069 (-14.8%) | 0.0225 | **0.1726** (+78.8%) | 0.0257 |
| **5. Hybrid MMR** ($\lambda=0.3$) | 0.0150 (-2.6%) | 0.0076 (-6.2%) | 0.0622 | 0.1168 | 0.0251 |
| **6. Score Blending** ($\lambda=0.3$) | 0.0148 (-3.9%) | 0.0074 (-8.6%) | 0.0229 | 0.0931 | **0.0259** |

### Empirical Results on WeChat (Full Validation Set, 19,737 Users):

| Re-Ranking Strategy | Recall@10 | NDCG@10 | Publisher Diversity@10 | Catalog Coverage@10 |
|---|---|---|---|---|
| **1. Pure Relevance** (Base) | 0.0023 | 0.0010 | 0.7869 | 39.71% |
| **2. Hard Creator Cap** (max 2/creator) | 0.0023 | 0.0011 | 0.9530 | 41.83% |
| **3. MMR Publisher ($\lambda=0.3$)** | **0.0030** (+30.4%) | **0.0013** (+30.0%) | **0.9623** (+22.3%) | **41.96%** (+5.7%) |
| **4. MMR Publisher ($\lambda=0.5$)** | **0.0033** (+43.5%) | **0.0014** (+40.0%) | **1.0000** (+27.1%) | **43.10%** (+8.5%) |

### Key Findings:
- **Win-Win Synergy Across Both Platforms:** MMR Publisher Diversification consistently improves both diversity and accuracy. On WeChat, where creator redundancy was high (with popular creators occupying up to 5 slots in a single user's top-10 list), MMR diversification eliminates creator monopolies, boosting **Publisher Diversity from 0.7869 to 1.0000 (+27.1%)** and **Recall@10 by +43.5%** (0.0023 → 0.0033).
- **Embedding vs. Publisher Diversity:** MMR in embedding space degrades accuracy (-13.0% Recall on TakaTak), indicating that latent collaborative vectors capture user interest clusters that should not be artificially dispersed, whereas publisher overconcentration represents redundant creator exposure.

---

## Research Ablation Study Matrix

The architecture includes eight systematic ablation variants (M1 through M8) to evaluate each paper's contribution:

| Variant | Architecture / Strategy | Adaptive Gate | 3-Entity CL | Freq BPR | Meta-Path | Soft Filter | MMR Div |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1** | LightGCN Baseline | – | – | – | – | – | – |
| **M2** | VA-GNN (Fixed Gate $\lambda=0.5$) | Fixed | – | – | – | – | – |
| **M3** | VA-GNN (Bilinear Adaptive Gate) | **Bilinear (Eq 17)** | – | – | – | – | – |
| **M4** | VA-GNN Core (+ 3-Entity CL) | **Bilinear** | **Yes** | – | – | – | – |
| **M5** | VA-GNN + Sagittarius WBPR | **Bilinear** | **Yes** | **Yes** | – | – | – |
| **M6** | VA-GNN + MPGCL Meta-Paths | **Bilinear** | **Yes** | **Yes** | **UVU + VUV** | – | – |
| **M7** | VA-GNN + Soft Filtering | **Bilinear** | **Yes** | **Yes** | – | **MLP Filter** | – |
| **M8** | Full Vynora + MMR Diversity | **Bilinear** | **Yes** | **Yes** | – | – | **$\lambda=0.3$** |

Run the full ablation matrix:

```bash
python ablation.py
```

---

## Limitations & Honest Reporting

1. **Embedding-Space vs. Semantic Diversity:** Content diversity is measured over learned collaborative latent representations ($e_v \in \mathbb{R}^d$), reflecting embedding-space dispersion rather than textual or visual semantic diversity (which would require video transcripts, tags, or frame embeddings).
2. **Leave-One-Out Ordering:** While the leave-one-out evaluation split is standard for implicit feedback benchmarks, interaction sequence order is not guaranteed chronological without explicit timestamps.
3. **Genre Metadata:** Genre diversity from Paper 5 is omitted because the underlying platforms do not supply genre taxonomies for micro-videos.

---

## Citation

If you build upon this work, please cite the foundational research:

- **VA-GNN**: Vlogger-Augmented Graph Neural Network for Micro-Video Recommendation
- **Sagittarius**: Behavior-aware Graph Convolution Network Model for Video Recommendation
- **MPGCL**: Meta-path Based Graph Contrastive Learning for Micro-Video Recommendation
- **CONDE**: Concept-Aware Denoising Graph Neural Network for Micro-Video Recommendation
- **Paper 5**: Chun et al., *"Predicting diversification scores of videos in recommendation network"*, Expert Systems with Applications, 238, 121803 (2024).
