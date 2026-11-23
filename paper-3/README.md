# Paper 3 Benchmark: MPGCL & MovieLens Adaptation

This directory contains everything related to benchmarking our framework against **Paper 3 (MPGCL)** on standard interaction datasets without a vlogger node (per Professor Dr. Amilpur Santhosh's directive: *"Add the third one also without vlogger"*).

It is governed by the project-wide **[ADR 0001 (Unified Multi-View Architecture)](../docs/adr/0001-unified-multiview-architecture.md)** and **[ADR 0003 (Full Catalog Ranking)](../docs/adr/0003-full-catalog-ranking-evaluation.md)**.

---

## 1. Paper 3 Details
* **Paper Title**: *MPGCL: Meta-path-based graph contrastive learning for micro-video recommendation*
* **Authors**: Yuan He, Cheng Wang, Jiahui Chen, Yanmin Chen, Charu C. Aggarwal
* **Venue**: *Expert Systems with Applications (ESWA)*, Volume 222, 2023, 119713.
* **Core Contribution of Paper 3**: 
  - Constructs meta-paths across heterogeneous graphs (e.g., $U \to I \to U$, $I \to U \to I$).
  - Projects them into type-specific homogeneous graphs.
  - Applies graph contrastive learning (NT-Xent) between augmented views.

---

## 2. Dataset Selection
Paper 3 evaluates on three datasets (Table 1 of Paper 3):
1. **MovieLens-1M (`MLs`)**: 6,040 users, 3,706 items, 1,000,209 interactions (95.53% sparsity).
   - **Properties**: Standard `{User, Item}` bipartite interaction dataset with **no vlogger node**.
   - **Local Path**: [`other-dataset's/movielens/ml-1m/ratings.dat`](file:///d:/study/college/5th%20semester/btp-v1/other-dataset's/movielens/ml-1m/ratings.dat).
2. **TikTok**: 3,656 users, 7,085 items, 1,253,112 interactions (95.51% sparsity).
3. **Kwai**: 169,878 users, 310,681 items, 775,834,643 interactions.

---

## 3. Official Published Baselines in Paper 3 for MovieLens-1M

### Table 5: Random Split (Train : Val : Test = 80% : 10% : 10%, K = 10, d = 200)
| Model | Type | Precision@10 | Recall@10 | NDCG@10 | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DeepFM** | Feature Factorization | 0.2323 | 0.3008 | 0.3991 | 0.6883 |
| **GraphSAGE** | Inductive GNN | 0.3172 | 0.3491 | 0.4100 | 0.7290 |
| **MMGCN** | Multimodal GNN | 0.3555 | 0.3889 | 0.4319 | 0.7383 |
| **LightGCN** | Simplified GCN | 0.3693 | 0.4198 | 0.4474 | 0.7499 |
| **UltraGCN** | Constraint-based GCN | 0.3782 | 0.4239 | 0.4599 | 0.7681 |
| **HAN** | Heterogeneous Attention | 0.3774 | 0.4295 | 0.4691 | 0.7661 |
| **HetGNN** | Heterogeneous GNN | 0.3812 | 0.4294 | 0.4784 | 0.7682 |
| **GCL** | Graph Contrastive | 0.3448 | 0.3994 | 0.4223 | 0.7094 |
| **HHFAN** | Heterogeneous Feature Attn | 0.3803 | 0.4405 | 0.4813 | 0.7790 |
| **MPGCL (Paper 3 SOTA)** | Meta-Path Contrastive | **0.4091** | **0.4636** | **0.5038** | **0.8121** |
| **Vynora (lambda=0.0)** | Denoised Meta-Path Contrastive | *Target > 0.4091* | *Target > 0.4636* | *Target > 0.5038* | *Target > 0.8121* |
| **Vynora (lambda=0.3)** | Denoised + Genre Diversity | *Target High* | *Target High* | *Target High* | *Target High* |

---

## 4. Architectural Innovation over Paper 3
Paper 3 constructs meta-paths ($U \to I \to U$ and $I \to U \to I$), but suffers because noisy implicit feedback contaminates the co-occurrence graphs. Vynora injects **CONDE-inspired iterative edge reliability pruning (EMA + dynamic 5% quantile pruning)** onto the interaction graph, eliminating false-positive edges before message passing and meta-path projection.

---

## 5. Quick Start (Google Colab with Auto-Resume Checkpointing)
1. Open Google Colab and upload [`paper3_movielens_benchmark.ipynb`](paper3_movielens_benchmark.ipynb).
2. Set Runtime to **T4 GPU**.
3. Run all cells.
4. **Auto-Resume**: If Colab disconnects, simply re-run the notebook. It detects `checkpoint_latest.pt` (saved on Google Drive) and automatically resumes training from the exact completed epoch!
