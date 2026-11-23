# 3. Full Catalog Ranking Evaluation Protocol

## Context
Initial WeChat and TakaTak benchmarks evaluated on a sampled candidate pool of 100 items (1 ground-truth positive + 99 unobserved items), whereas standard literature benchmarks (including Paper 3 on MovieLens-1M) rank over the full item catalog. In modern RecSys research (e.g., Krichene & Rendle, ACM RecSys), sampled metrics are heavily criticized for metric instability and unfair baseline comparisons.

## Decision
We standardize on Full Catalog Ranking across all three datasets (WeChat, TakaTak, MovieLens) as our primary reported benchmark. Candidate pool sampling (Top-100) is retained strictly as an efficiency ablation.

## Consequences
- Evaluation metrics are 100% immune to reviewer critique regarding negative sampling bias.
- Directly comparable to published literature tables in Paper 3 (MPGCL) and Paper 4 (VA-GNN).
