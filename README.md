# Vynora 2.0 — Denoising & Vlogger-Aware Meta-Path Graph CL with Diversity Recommendation

A research implementation for micro-video recommendation that synthesizes four foundational papers:

| Paper | What we take |
|---|---|
| **VA-GNN** (Paper 4) | Tripartite graph (User, Video, Vlogger), dual-view propagation, lambda-weighted scoring |
| **MPGCL** (Paper 3) | Meta-path views, cross-view InfoNCE contrastive loss, ProjectionHead |
| **CONDE** (Paper 2) | Edge reliability scoring, EMA score accumulation, periodic graph denoising |
| **Paper 5** (Chun et al., ESWA 2024) | Diversification score prediction (GNN with feature-augmented attention), publisher & content diversity metrics (Eq 1 & 2), and diversity re-ranking (MMR & Score Blending) |

---

## Project Structure

```
vynora2/
├── config.py                          ← Hyperparameters, paths & diversity settings
├── dataloader.py                      ← VynoraDataset: loads takatak/wechat, builds sparse graphs
├── model.py                           ← VynoraGNN: dual-view GNN + contrastive + edge scorer
├── sampler.py                         ← BPR dual-stream negative sampler (VA-GNN pattern)
├── denoiser.py                        ← CONDE-style iterative graph denoiser
├── diversity.py                       ← Paper 5 implementation: diversification GNN, metrics, re-ranking
├── train_diversification_predictor.py ← Trains Paper 5 GNN to predict diversification scores S_i
├── diversity_benchmark.py             ← Accuracy vs. Diversity Pareto trade-off evaluator
├── evaluate.py                        ← Recall@K, NDCG@K, Precision@K, MRR@K + Diversity@K
├── train.py                           ← Full training loop (run this)
├── predict.py                         ← Home screen inference with Paper 5 diversity controls
├── ablation.py                        ← Six-variant ablation study (A–F)
├── checkpoints/                       ← Saved model weights
└── results/                           ← JSON benchmark results and ablation summaries
```

---

## Data

Uses the VA-GNN dataset format from `VAgnn-dataset/` or `other-dataset's/`:

```
data_size.txt            n_users  n_videos  n_vloggers
user_video_data.txt      user_id  v1 v2 ... vN  (last 2 = val, test)
user_vlogger_data.txt    user_id  a1 a2 ... aN  (last 2 = val, test)
user_vlogger_data2.txt   user_id  a1 a2 ... aN  (auxiliary vlogger stream)
vlogger_video_data.txt   vlogger_id  video_id   (tab-separated pairs)
```

| Dataset   | Users  | Videos | Vloggers |
|-----------|--------|--------|----------|
| takatak   | 14,571 | 13,133 |    912   |
| wechat    | 19,737 | 25,976 |  2,088   |

---

## Setup

```bash
pip install torch scipy numpy tqdm
```

No PyTorch Geometric or DGL required — all graph operations use native PyTorch sparse tensors.

---

## Paper 5 Diversity Recommendation

Without diversity control, recommendations often suffer from severe **creator / publisher redundancy** (e.g. 5 out of 10 recommended videos coming from the single same viral vlogger).

Following **Paper 5** (*"Predicting diversification scores of videos in recommendation network"*, Chun et al., ESWA 2024), Vynora 2.0 incorporates:

### 1. Diversity Formulas (Paper 5 Eq 1 & 2)
For a top-$n$ recommendation list $L$:
$$\text{Diversity}(L) = \frac{\sum_{i, j \in L, i \neq j} d(i, j)}{n \cdot (n - 1)}$$

- **Publisher Diversity**:
  $$d_{pub}(i, j) = \begin{cases} 1 & \text{if } \text{vlogger}(i) \neq \text{vlogger}(j) \\ 0 & \text{otherwise} \end{cases}$$
  Efficiently computed via Gini-Simpson distance: $\text{Div}_{pub} = 1 - \frac{\sum_v c_v(c_v - 1)}{n(n - 1)}$.

- **Content Diversity**:
  $$d_{con}(i, j) = 1 - \text{CosineSimilarity}(\mathbf{e}_i, \mathbf{e}_j)$$
  Computed in $O(n \cdot D)$ time: $\text{Div}_{con} = 1 - \frac{\|\sum_{i=1}^n \hat{\mathbf{e}}_i\|^2 - n}{n(n - 1)}$.

### 2. Diversification Scores ($S_i$)
Each video $i$ possesses an empirical diversification score defined in Paper 5 Section 3 as the average diversity experienced by all users who watched video $i$:
$$S_i = \frac{1}{|U_i|} \sum_{u \in U_i} \text{Diversity}(u)$$

### 3. Diversification Score Prediction GNN (Paper 5 Section 4.5)
A graph neural network with feature-augmented attention:
- Node feature embedding: $\mathbf{h}_i^{(0)} = \mathbf{W}_{embed} \mathbf{r}_i$
- 4-dim Edge features: $\mathbf{h}_e = [\text{freq}_{ij}, \text{rank}_{ij}^{-1}, d(i, j), 1 - d(i, j)]$
- Attention: $\alpha_{j, i, e} = \text{softmax}_j\left(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}_N \mathbf{h}_j \,\|\, \mathbf{W}_N \mathbf{h}_i \,\|\, \mathbf{W}_E \mathbf{h}_e])\right)$
- Prediction MLP: $\hat{S}_i = \mathbf{W}_2(\sigma(\mathbf{W}_1 \mathbf{h}_i))$ trained with $\mathcal{L}_{MAE}$.

```bash
# Train the Paper 5 GNN diversification predictor
python train_diversification_predictor.py --dataset takatak --epochs 20
```

### 4. Diversity-Aware Inference & Re-Ranking (Paper 5 Section 7)
Run inference with custom diversity trade-off $\lambda \in [0.0, 1.0]$:

```bash
# Balanced diversity recommendation (lambda=0.4, MMR)
python predict.py --user 42 --topk 10 --dataset takatak --diversity 0.4

# Pure relevance baseline (lambda=0.0)
python predict.py --user 42 --topk 10 --dataset takatak --diversity 0.0

# Using Paper 5 Score Blending Re-ranking
python predict.py --user 42 --topk 10 --diversity 0.4 --diversity_method score_rerank
```

#### Example Output on User 42:
```
========================================================
  Home Screen: Top-10 for User 42  (takatak)
  Mode: Paper 5 Diversity (lambda=0.4, publisher, mmr)
========================================================
Rank  Video ID   Vlogger     Score
------------------------------------
   1       284        64    6.0992
   2        43        36    5.3996
   3         2         2    5.3651
   4       109        12    5.1991
   5       878        29    5.1313
   6       287         9    5.0414
   7       978        18    4.9253
   8      1621        38    4.9104
   9         4         4    4.8444
  10         6         6    4.8263
------------------------------------

Paper 5 Diversity Analysis (Top-10):
  - Publisher Diversity (Eq 1):  1.0000
  - Content Diversity (Eq 1 & 2): 0.0132
  - Unique Vloggers:             10 / 10
  - Vlogger Shannon Entropy:     3.3219 bits

Diversity Improvement vs Baseline:
  - Publisher Diversity Gain:    +28.6%  (0.7778 -> 1.0000)
  - Creator Variety Increase:    +4 unique vloggers (6 -> 10)
```

---

## Accuracy vs. Diversity Benchmark

Run the full trade-off evaluation over validation or test sets:

```bash
python diversity_benchmark.py --dataset takatak --mode val
```

### Experimental Results on Takatak (Val Set):
| $\lambda_{div}$ | Recall@10 | NDCG@10 | Publisher Diversity@10 | Content Diversity@10 | Catalog Coverage@10 |
|---|---|---|---|---|---|
| **0.00** (Base) | 0.0436 | 0.0216 | 0.8424 | 0.0180 | 0.0222 |
| **0.30** (Paper 5) | **0.0472** | **0.0220** | **0.9928** | 0.0151 | **0.0228** |

> [!NOTE]
> Diversification at $\lambda = 0.3$ not only increases Publisher Diversity from **0.8424 to 0.9928** (+17.8%), it also **improves Recall@10 by +8.2%** (from 0.0436 to 0.0472). By preventing redundant recommendations from the same vlogger, the recommender covers the user's multi-faceted interests and recovers true watched items more effectively.

---

## Ablation Study

Reproduces the six-variant experimental design from the research plan:

| Variant | L_rank | Vlogger | CL | Denoising |
|---------|--------|---------|-----|-----------|
| A. Baseline | ✓ | – | – | – |
| B. + Vlogger | ✓ | ✓ | – | – |
| C. + CL | ✓ | ✓ | ✓ | – |
| D. + Denoising | ✓ | – | – | ✓ |
| E. Vlogger + Denoise | ✓ | ✓ | – | ✓ |
| F. Full model | ✓ | ✓ | ✓ | ✓ |

```bash
python ablation.py
```
