# 4. Type Diversity Formulation

## Context
Recommendation diversity in our project originated as Publisher Diversity (Gini-Simpson over creators in micro-video feeds) to break winner-take-all creator monopolies. On datasets without creator metadata (such as MovieLens), diversity must still be enforced to prevent algorithmic echo chambers.

## Decision
We define a unified, domain-agnostic metric: **Type Diversity**:
$$D(L) = 1 - \frac{\sum_c n_c(n_c - 1)}{K(K - 1)}$$
where $c$ is any discrete item attribute partition: Content Creator/Publisher for micro-video feeds, Genre for movies, and Topic for articles.

## Consequences
- The diversification mechanism is proved domain-agnostic and universally applicable.
- In all datasets, a controllable parameter $\lambda_{\text{div}} \in [0, 1]$ allows switching between pure accuracy ($\lambda = 0.0$ to compare against standard accuracy baselines) and high-diversity feeds ($\lambda > 0$).
