# 1. Unified Multi-View Architecture

## Context
Vynora was initially developed for creator-dense micro-video platforms (WeChat Channels and TakaTak) utilizing a tripartite graph {User, Video, Creator}. However, academic benchmarking requires evaluating on standard benchmark datasets (such as MovieLens-1M) where explicit creator/vlogger metadata is absent.

## Decision
We unify the framework under a single mathematical formulation:
$$V = \lambda_v \mathcal{G}_{uv} + \lambda_a \mathcal{G}_{ua} + \lambda_{meta} \mathcal{G}_{meta}$$
where the creator stream $\mathcal{G}_{ua}$ is an optional view. When creator metadata is present (WeChat, TakaTak), the full tripartite model is active. When creator metadata is absent (MovieLens), $\lambda_a = 0$ and the model seamlessly operates as a denoised bipartite GNN with meta-path contrastive learning without code bifurcation or ad-hoc architectural patches.

## Consequences
- Single unified paper narrative: Vynora is presented as a universally applicable multi-view framework.
- Zero code fragmentation across datasets.
