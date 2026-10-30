"""
model.py  –  Vynora 2.0
Denoising & Vlogger-Aware Meta-Path Graph Contrastive Learning

Architecture synthesises three papers:
  • VA-GNN  : tripartite graph (User, Video, Vlogger) + dual-view propagation
  • MPGCL   : meta-path graph views + cross-view contrastive learning (InfoNCE)
  • CONDE   : edge reliability scoring → iterative graph denoising

Key design decisions drawn from CONDE source code (Graph-Denoising-SocialRec):
  - Edge reliability is a LEARNED scalar per edge, accumulated across epochs
    (CONDE: relation_score_all_epoch = beta*old + (1-beta)*new, then drop lowest)
  - Denoising happens at epoch boundaries, not every forward pass
  - The GNN itself uses layer-aggregation concatenation (LightGCN-style)

Key design decisions drawn from VA-GNN source code:
  - Separate embedding tables for users, videos, vloggers
  - Two adjacency matrices: User-Video (video-view) and User-Vlogger (vlogger-view)
  - BPR loss with dual positive/negative streams
  - Lambda-weighted scoring: s = lambda*s_video + (1-lambda)*s_vlogger

Key design decisions drawn from MPGCL:
  - ProjectionHead before InfoNCE (GELU + residual + LayerNorm)
  - Symmetric InfoNCE loss across the two graph views
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ═══════════════════════════════════════════════════════════════
# Projection Head  (shared between both graph views)
# ═══════════════════════════════════════════════════════════════
class ProjectionHead(nn.Module):
    """
    MPGCL-style projection head: Linear → GELU → Linear → Dropout → Residual → LayerNorm
    Maps node embeddings into contrastive comparison space.
    """
    def __init__(self, embed_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.proj   = nn.Linear(embed_dim, proj_dim)
        self.gelu   = nn.GELU()
        self.fc     = nn.Linear(proj_dim, proj_dim)
        self.drop   = nn.Dropout(dropout)
        self.norm   = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected  = self.proj(x)
        out        = self.gelu(projected)
        out        = self.fc(out)
        out        = self.drop(out)
        out        = out + projected        # residual
        return self.norm(out)


# ═══════════════════════════════════════════════════════════════
# Edge Reliability Scorer  (CONDE-inspired)
# ═══════════════════════════════════════════════════════════════
class EdgeReliabilityScorer(nn.Module):
    """
    Scores the reliability of a (user, neighbor) edge using their learned embeddings.
    Output is a scalar in [0, 1] for each edge.

    Adapted from CONDE's TransformerEncoder relation scorer:
    Instead of a full Transformer over visited-item sequences (which requires
    the full item history matrix), we use a lightweight MLP on concatenated
    node pairs — appropriate for the VA-GNN data format.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, emb_src: torch.Tensor, emb_dst: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_src: [E, D] source node embeddings
            emb_dst: [E, D] destination node embeddings
        Returns:
            scores:  [E]   reliability score per edge
        """
        return self.scorer(torch.cat([emb_src, emb_dst], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# LightGCN-style propagation layer (no non-linearity, no weights)
# ═══════════════════════════════════════════════════════════════
def light_gcn_propagate(adj: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """
    Single LightGCN propagation step: E_out = A_norm @ E_in
    adj is a (n x n) normalised sparse torch tensor.
    emb is a (n x d) dense embedding matrix.
    """
    return torch.sparse.mm(adj, emb)


# ═══════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════
class VynoraGNN(nn.Module):
    """
    Vynora 2.0 – Denoising & Vlogger-Aware Meta-Path Graph Contrastive Learning.

    Two-view propagation:
      view_v  (video-view)   : propagates over User-Video adjacency
      view_a  (vlogger-view) : propagates over User-Vlogger adjacency

    Graph denoising (CONDE-style, applied externally between epochs):
      The caller maintains a running reliability score per edge and passes
      masked/pruned adjacency matrices after every D epochs.

    Loss:
      L = L_bpr_uv + beta * L_cl + gamma * L_den + delta * L_bpr_ua
    """

    def __init__(self, n_users: int, n_videos: int, n_vloggers: int):
        super().__init__()

        self.n_users   = n_users
        self.n_videos  = n_videos
        self.n_vloggers = n_vloggers
        D = config.EMBED_DIM

        # ── Embedding tables ──────────────────────────────────────────
        self.emb_user    = nn.Embedding(n_users,    D)
        self.emb_video   = nn.Embedding(n_videos,   D)
        self.emb_vlogger = nn.Embedding(n_vloggers, D)
        nn.init.normal_(self.emb_user.weight,    std=0.1)
        nn.init.normal_(self.emb_video.weight,   std=0.1)
        nn.init.normal_(self.emb_vlogger.weight, std=0.1)

        # ── Projection heads (one per view, MPGCL design) ─────────────
        L = config.N_LAYERS
        concat_dim = D * (L + 1)   # layer-aggregation output size
        self.proj_v = ProjectionHead(concat_dim, config.PROJECTION_DIM, config.PROJECTION_DROP)
        self.proj_a = ProjectionHead(concat_dim, config.PROJECTION_DIM, config.PROJECTION_DROP)

        # ── Edge reliability scorer (CONDE-inspired) ───────────────────
        self.edge_scorer_uv = EdgeReliabilityScorer(D)
        self.edge_scorer_ua = EdgeReliabilityScorer(D)

        # ── Scoring lambda ─────────────────────────────────────────────
        self.lambda_vlogger = config.LAMBDA_VLOGGER

    # ────────────────────────────────────────────────────────────────
    # Forward – returns all embeddings needed for loss computation
    # ────────────────────────────────────────────────────────────────
    def forward(self, graph_uv: torch.Tensor, graph_ua: torch.Tensor):
        """
        Args:
            graph_uv: D^{-½}AD^{-½} sparse (n_users + n_videos) x (n_users + n_videos)
            graph_ua: D^{-½}AD^{-½} sparse (n_users + n_vloggers) x (n_users + n_vloggers)

        Returns dict with keys:
            user_v, video_v   : video-view final embeddings  [N, D*(L+1)]
            user_a, vlogger_a : vlogger-view final embeddings
            user_v0, user_a0  : 0th-layer user embeddings (for contrastive)
        """
        # ── Video-view propagation ────────────────────────────────────
        e_user_v  = self.emb_user.weight
        e_video   = self.emb_video.weight
        emb_v     = torch.cat([e_user_v, e_video], dim=0)    # [nU+nV, D]
        layers_v  = [emb_v]
        for _ in range(config.N_LAYERS):
            emb_v = light_gcn_propagate(graph_uv, emb_v)
            layers_v.append(emb_v)
        out_v   = torch.cat(layers_v, dim=-1)                 # [nU+nV, D*(L+1)]
        user_v  = out_v[:self.n_users]
        video_v = out_v[self.n_users:]

        # ── Vlogger-view propagation ──────────────────────────────────
        e_user_a  = self.emb_user.weight
        e_vlogger = self.emb_vlogger.weight
        emb_a     = torch.cat([e_user_a, e_vlogger], dim=0)  # [nU+nP, D]
        layers_a  = [emb_a]
        for _ in range(config.N_LAYERS):
            emb_a = light_gcn_propagate(graph_ua, emb_a)
            layers_a.append(emb_a)
        out_a     = torch.cat(layers_a, dim=-1)               # [nU+nP, D*(L+1)]
        user_a    = out_a[:self.n_users]
        vlogger_a = out_a[self.n_users:]

        return {
            "user_v":    user_v,
            "video_v":   video_v,
            "user_a":    user_a,
            "vlogger_a": vlogger_a,
        }

    # ────────────────────────────────────────────────────────────────
    # Scoring function  (VA-GNN eq.)
    # ────────────────────────────────────────────────────────────────
    def score(self, users, videos, vlogger_ids,
              user_v, video_v, user_a, vlogger_a):
        """
        s(u, v) = lambda * (u_v · v_v) + (1-lambda) * (u_a · p_a)
        """
        u_v = user_v[users]    # [B, D*(L+1)]
        v_v = video_v[videos]
        u_a = user_a[users]
        p_a = vlogger_a[vlogger_ids]

        s_video   = (u_v * v_v).sum(dim=-1)
        s_vlogger = (u_a * p_a).sum(dim=-1)
        return self.lambda_vlogger * s_video + (1 - self.lambda_vlogger) * s_vlogger

    # ────────────────────────────────────────────────────────────────
    # Loss functions
    # ────────────────────────────────────────────────────────────────
    def bpr_loss_uv(self, users, pos_videos, neg_videos, pos_vloggers, neg_vloggers,
                    user_v, video_v, user_a, vlogger_a):
        """
        BPR loss for user-video recommendation (primary task).
        Positive: (user, pos_video, vlogger_of_pos_video)
        Negative: (user, neg_video, vlogger_of_neg_video)
        """
        s_pos = self.score(users, pos_videos, pos_vloggers, user_v, video_v, user_a, vlogger_a)
        s_neg = self.score(users, neg_videos, neg_vloggers, user_v, video_v, user_a, vlogger_a)
        loss  = -F.logsigmoid(s_pos - s_neg).mean()

        # L2 regularisation on initial embeddings only (weight_decay in optimizer)
        reg = (self.emb_user.weight[users].norm(2).pow(2) +
               self.emb_video.weight[pos_videos].norm(2).pow(2) +
               self.emb_video.weight[neg_videos].norm(2).pow(2)) / (2 * len(users))
        return loss + config.WEIGHT_DECAY * reg

    def bpr_loss_ua(self, users2, pos_vloggers2, neg_vloggers2, user_a, vlogger_a):
        """
        Auxiliary BPR loss for user-vlogger affinity (delta * L_vlogger, VA-GNN eq.).
        """
        u  = user_a[users2]
        pa = vlogger_a[pos_vloggers2]
        na = vlogger_a[neg_vloggers2]
        s_pos = (u * pa).sum(dim=-1)
        s_neg = (u * na).sum(dim=-1)
        return -F.logsigmoid(s_pos - s_neg).mean()

    def contrastive_loss(self, user_v, user_a, batch_users):
        """
        Cross-view InfoNCE loss (MPGCL-style).
        Aligns user representations from video-view and vlogger-view.
        Positives: same user across the two views.
        Negatives: all other users in the batch.
        """
        z_v = F.normalize(self.proj_v(user_v[batch_users]), dim=-1)   # [B, proj_dim]
        z_a = F.normalize(self.proj_a(user_a[batch_users]), dim=-1)   # [B, proj_dim]

        tau  = config.TEMPERATURE
        # Compute similarity matrix [B, B]
        sim  = torch.matmul(z_v, z_a.T) / tau
        # Symmetric InfoNCE
        B    = sim.size(0)
        labels = torch.arange(B, device=sim.device)
        loss_v = F.cross_entropy(sim,   labels)
        loss_a = F.cross_entropy(sim.T, labels)
        return (loss_v + loss_a) / 2

    def denoising_consistency_loss(self, user_v, user_a, batch_users):
        """
        Auxiliary denoising loss: the two views of the SAME user should
        remain close even after edge removal. We use MSE on L2-normalised
        embeddings as a soft consistency regulariser (gamma * L_den).
        """
        z_v = F.normalize(user_v[batch_users], dim=-1)
        z_a = F.normalize(user_a[batch_users], dim=-1)
        return F.mse_loss(z_v, z_a)

    # ────────────────────────────────────────────────────────────────
    # Edge reliability scoring  (CONDE denoising interface)
    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def compute_edge_reliability_uv(self, src_users, dst_videos):
        """
        Score user-video edges using the current embedding state.
        Called externally after each epoch; scores are accumulated by trainer.
        Returns: [E] tensor of reliability scores in [0,1].
        """
        eu = self.emb_user.weight[src_users]
        ev = self.emb_video.weight[dst_videos]
        return self.edge_scorer_uv(eu, ev)

    @torch.no_grad()
    def compute_edge_reliability_ua(self, src_users, dst_vloggers):
        """Score user-vlogger edges."""
        eu = self.emb_user.weight[src_users]
        ea = self.emb_vlogger.weight[dst_vloggers]
        return self.edge_scorer_ua(eu, ea)

    # ────────────────────────────────────────────────────────────────
    # getUsersRating – vectorised scoring for evaluation
    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def getUsersRating(self, users, graph_uv, graph_ua):
        """
        Returns a [|users|, n_videos] score matrix for evaluation.
        """
        emb = self.forward(graph_uv, graph_ua)
        u_v = emb["user_v"][users]          # [B, D*(L+1)]
        v_v = emb["video_v"]                # [n_videos, D*(L+1)]
        u_a = emb["user_a"][users]          # [B, D*(L+1)]
        p_a = emb["vlogger_a"]              # [n_vloggers, D*(L+1)]

        # Expand vlogger embeddings indexed by video
        s_video   = torch.matmul(u_v, v_v.T)   # [B, n_videos]

        # For every video, its vlogger's embedding  [n_videos, D*(L+1)]
        # (efficient: avoid loop by matrix multiply then index)
        # s_vlogger[b,v] = u_a[b] · p_a[vlogger_of_v]
        # Build per-video vlogger embedding matrix
        #   This is pre-computed once and cached in the trainer.
        # Here we return s_video only if vlogger_of_video tensor not passed.
        return s_video    # caller adds vlogger score if needed
