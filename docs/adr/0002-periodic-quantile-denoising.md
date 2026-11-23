# 2. Periodic Quantile Denoising with EMA

## Context
Two competing graph denoising schemes were present in the codebase: continuous differentiable edge weighting during forward pass (`SoftInteractionFilter`) vs. discrete periodic quantile edge pruning with Exponential Moving Average (EMA) accumulation (`BipartiteGraphDenoiser` / `CONDE` adaptation).

## Decision
We formally adopt Periodic Quantile Graph Denoising with EMA as our sole denoising mechanism, running every $D = 5$ epochs to prune the bottom 5% quantile unreliable edges. Unused soft-filtering modules are deprecated.

## Consequences
- Physical sparsification of the graph adjacency matrix prevents over-smoothing and accelerates sparse matrix multiplication.
- EMA smoothing ensures that newly explored or transient edges are not prematurely pruned.
- Direct experimental backing: this exact mechanism generated our verified +185% Recall improvement on WeChat.
