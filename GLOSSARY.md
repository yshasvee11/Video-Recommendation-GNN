# Vynora Domain Glossary

Core domain terminology and structural definitions for the Vynora recommendation framework.

## Graph Modeling

**Interaction Graph**:
The bipartite or tripartite graph representing observed user engagements with items and creators.
_Avoid_: Bipartite matrix, user-item network

**Meta-Path Homogeneous View**:
A projected single-entity-type graph derived from multi-hop composite relations (e.g., User-Item-User or Item-User-Item) capturing indirect structural affinity.
_Avoid_: Auxiliary subgraph, projected network

**Graph Denoising**:
The iterative process of evaluating edge reliability via latent representation affinity and pruning spurious or accidental interaction edges.
_Avoid_: Edge filtering, graph cleaning, pruning mask

## Diversification & Ranking

**Type Diversity**:
The Gini-Simpson measure of distributional dispersion across an item category attribute (publisher, genre, or topic) within a Top-K recommendation list.
_Avoid_: Intra-list diversity, category dispersion

**Publisher Diversity**:
The specific instantiation of Type Diversity measuring the spread of content creators in a Top-K micro-video recommendation list.
_Avoid_: Vlogger diversity, creator entropy

**Hard Creator Capping**:
A deterministic ranking constraint restricting any single content creator to at most a fixed maximum number of slots (e.g., $\le 2$) in a user's Top-K feed.
_Avoid_: Creator threshold, publisher limit

**Empirical Diversification Score**:
The historical average diversity experienced by all users who interacted with a given item in training data, used as a prior for diversity re-ranking.
_Avoid_: Diversity label, diversity prior
