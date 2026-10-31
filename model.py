"""
model.py  –  Vynora 2.0
Creator-Aware Dual-View GNN with Cross-View Contrastive Learning and Edge Pruning.

HONEST ARCHITECTURE NOTES
--------------------------
This model is INSPIRED by three papers, but is NOT a faithful reproduction of any:

VA-GNN inspiration (Paper 4):
  - We use the tripartite graph (User, Video, Vlogger) and dual-view propagation.
  - We use a lambda-weighted scoring: s = lambda * s_video + (1 - lambda) * s_vlogger.
  - DIFFERENCE: VA-GNN learns an adaptive gate conditioned on each candidate video and
    its creator. We use a FIXED lambda (config.LAMBDA_VLOGGER = 0.5). This means our
    scoring mixture is not data-driven. The creator-video relation is also not yet
    incorporated as a separate propagation path during message passing.

MPGCL inspiration (Paper 3):
  - We use a cross-view InfoNCE contrastive loss with a ProjectionHead (GELU + residual
    + LayerNorm), which is structurally similar to MPGCL's contrastive component.
  - DIFFERENCE: MPGCL constructs META-PATH-DERIVED HOMOGENEOUS graphs (e.g.,
    User-Video-User, User-Vlogger-User meta-paths) as separate contrastive views.
    We contrast the two DIRECT INTERACTION views (User-Video bipartite vs User-Vlogger
    bipartite). This is a simpler dual-view contrast, not MPGCL meta-path graphs.

CONDE inspiration (Paper 2):
  - We implement an edge reliability scorer, EMA score accumulation across epochs,
    and the logarithmic drop schedule (log2(|N(u)|)^gamma * R).
  - DIFFERENCE: CONDE uses a TransformerEncoder over a user's full item-visit sequence
    and concept-aware personalized selection. Our scorer is a lightweight MLP on
    concatenated (user_emb, item_emb) pairs. It is NOT trained end-to-end through the
    recommendation loss (edge scores are computed post-forward, via a separate
    compute_edge_reliability_* call), so the pruning signal is weaker than CONDE's.
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
    Projection head for cross-view contrastive learning.
    Architecture: Linear -> GELU -> Linear -> Dropout -> Residual -> LayerNorm

    INSPIRATION: MPGCL uses this design before InfoNCE.
    NOTE: MPGCL applies this to meta-path homogeneous graph views.
    Here it is applied to the two direct interaction views (User-Video, User-Vlogger).
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
# Soft Interaction Filtering (CONDE-Inspired Adaptation)
# ═══════════════════════════════════════════════════════════════
class SoftInteractionFilter(nn.Module):
    """
    Differentiable soft relevance filtering (CONDE adaptation).
    Replaces destructive discrete edge dropping with an end-to-end relevance attention:
      r_{uv} = MLP([e_u || e_v || e_u * e_v]) -> scalar in (0, 1)
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, u_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([u_emb, v_emb, u_emb * v_emb], dim=-1)
        return self.mlp(cat).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Edge Reliability Scorer  (Legacy CONDE-inspired pruning)
# ═══════════════════════════════════════════════════════════════
class EdgeReliabilityScorer(nn.Module):
    """
    Lightweight MLP that scores the reliability of a (user, neighbor) edge using
    their learned embeddings. Output is a scalar in [0, 1] per edge.
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
        return self.scorer(torch.cat([emb_src, emb_dst], dim=-1)).squeeze(-1)
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
    Vynora – Creator-Aware Dual-View GNN with Cross-View Contrastive Learning.

    Two-view propagation (VA-GNN inspired):
      view_v  (video-view)   : propagates over User-Video adjacency
      view_a  (vlogger-view) : propagates over User-Vlogger adjacency

    Edge pruning (CONDE-inspired, applied externally between epochs):
      The caller accumulates EMA reliability scores per edge and passes
      pruned adjacency matrices after every D epochs.

    Score mixing (VA-GNN pattern, simplified):
      s(u, v) = lambda * (u_v . v_v) + (1-lambda) * (u_a . p_a)
      NOTE: lambda is FIXED at config.LAMBDA_VLOGGER. VA-GNN learns an adaptive
      gate conditioned on each candidate video and its creator.

    Total loss:
      L = L_bpr_uv + beta * L_cl + gamma * L_den + delta * L_bpr_ua
    """

    def __init__(self, n_users: int, n_videos: int, n_vloggers: int, vlogger_list=None):
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

        # ── VA-GNN Bilinear Adaptive Gate (Eq. 17): w = sigmoid(e_v^T Q e_{p_v}) ──
        # Learnable bilinear parameter matrix matching the concatenated embedding dimension
        self.gate_Q = nn.Parameter(torch.empty(concat_dim, concat_dim))
        nn.init.xavier_uniform_(self.gate_Q.data, gain=1.0)

        # ── Soft Interaction Filter (CONDE adaptation) ────────────────
        self.soft_filter = SoftInteractionFilter(D)

        # ── Legacy Edge reliability scorer (CONDE-inspired) ───────────
        self.edge_scorer_uv = EdgeReliabilityScorer(D)
        self.edge_scorer_ua = EdgeReliabilityScorer(D)

        # ── Fallback scoring lambda (if adaptive gate disabled in ablation) ─
        self.lambda_vlogger = config.LAMBDA_VLOGGER

        # Register vlogger_list if provided
        if vlogger_list is not None:
            if isinstance(vlogger_list, torch.Tensor):
                self.register_buffer("vlogger_list", vlogger_list.clone().detach().long())
            else:
                self.register_buffer("vlogger_list", torch.LongTensor(vlogger_list))
        else:
            self.vlogger_list = None

    # ────────────────────────────────────────────────────────────────
    # VA-GNN Bilinear Adaptive Gate Equation 17
    # ────────────────────────────────────────────────────────────────
    def compute_adaptive_gate(self, v_emb: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        """
        VA-GNN Equation 17:
          w = sigmoid(e_v^T Q e_{p_v})
        Args:
            v_emb: [*, concat_dim] video representation
            p_emb: [*, concat_dim] creator representation
        Returns:
            w: [*] adaptive balance weight in (0, 1)
        """
        v_proj = torch.matmul(v_emb, self.gate_Q)
        logits = (v_proj * p_emb).sum(dim=-1)
        return torch.sigmoid(logits)

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
    # Scoring function  (VA-GNN Bilinear Adaptive Gate)
    # ────────────────────────────────────────────────────────────────
    def score(self, users, videos, vlogger_ids,
              user_v, video_v, user_a, vlogger_a):
        """
        VA-GNN Bilinear Adaptive Scoring:
          s(u, v) = w_{uv} * (u_v · v_v) + (1 - w_{uv}) * (u_a · p_a)
          where w_{uv} = sigmoid(v_v^T Q p_a) if USE_ADAPTIVE_GATE else config.LAMBDA_VLOGGER
        """
        u_v = user_v[users]    # [B, D*(L+1)]
        v_v = video_v[videos]
        u_a = user_a[users]
        p_a = vlogger_a[vlogger_ids]

        s_video   = (u_v * v_v).sum(dim=-1)
        s_vlogger = (u_a * p_a).sum(dim=-1)

        if getattr(config, "USE_ADAPTIVE_GATE", True):
            w = self.compute_adaptive_gate(v_v, p_a)
            return w * s_video + (1.0 - w) * s_vlogger
        else:
            return self.lambda_vlogger * s_video + (1.0 - self.lambda_vlogger) * s_vlogger

    # ────────────────────────────────────────────────────────────────
    # Loss functions
    # ────────────────────────────────────────────────────────────────
    def bpr_loss_uv(self, users, pos_videos, neg_videos, pos_vloggers, neg_vloggers,
                    user_v, video_v, user_a, vlogger_a, weights=None):
        """
        BPR loss for user-video recommendation (primary task).
        Supports Sagittarius frequency weighting w_{uv} = log(1 + c_{uv}).
        """
        s_pos = self.score(users, pos_videos, pos_vloggers, user_v, video_v, user_a, vlogger_a)
        s_neg = self.score(users, neg_videos, neg_vloggers, user_v, video_v, user_a, vlogger_a)
        diff = s_pos - s_neg

        if weights is not None and getattr(config, "USE_FREQUENCY_WEIGHTING", True):
            loss = -(weights * F.logsigmoid(diff)).mean()
        else:
            loss = -F.logsigmoid(diff).mean()

        # L2 regularisation on initial embeddings only
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

    def contrastive_loss(self, user_v, user_a, batch_users,
                         video_v=None, vlogger_a=None,
                         batch_videos=None, batch_vloggers=None):
        """
        VA-GNN 3-Entity Cross-View Contrastive Loss:
          L_CL = (L_user + L_video + L_creator) / 3
        Aligns representations across video-view and vlogger-view for users,
        videos, and creators.
        """
        tau = config.TEMPERATURE

        # 1. User contrast
        z_v = F.normalize(self.proj_v(user_v[batch_users]), dim=-1)   # [B, proj_dim]
        z_a = F.normalize(self.proj_a(user_a[batch_users]), dim=-1)   # [B, proj_dim]

        sim_u  = torch.matmul(z_v, z_a.T) / tau
        labels = torch.arange(len(batch_users), device=sim_u.device)
        loss_u = (F.cross_entropy(sim_u, labels) + F.cross_entropy(sim_u.T, labels)) / 2.0

        if not getattr(config, "USE_3ENTITY_CL", True) or video_v is None or batch_videos is None:
            return loss_u

        # 2. Video contrast (align candidate video with creator representation)
        u_vids = torch.unique(batch_videos)
        if len(u_vids) > 1 and self.vlogger_list is not None:
            v_creators = self.vlogger_list[u_vids]
            z_vid_v = F.normalize(self.proj_v(video_v[u_vids]), dim=-1)
            z_vid_a = F.normalize(self.proj_a(vlogger_a[v_creators]), dim=-1)
            sim_v = torch.matmul(z_vid_v, z_vid_a.T) / tau
            lbl_v = torch.arange(len(u_vids), device=sim_v.device)
            loss_v = (F.cross_entropy(sim_v, lbl_v) + F.cross_entropy(sim_v.T, lbl_v)) / 2.0
        else:
            loss_v = loss_u

        # 3. Creator contrast
        if batch_vloggers is not None:
            u_vlogs = torch.unique(batch_vloggers)
            if len(u_vlogs) > 1:
                z_crt_a = F.normalize(self.proj_a(vlogger_a[u_vlogs]), dim=-1)
                z_crt_v = F.normalize(self.proj_v(vlogger_a[u_vlogs]), dim=-1)
                sim_p = torch.matmul(z_crt_a, z_crt_v.T) / tau
                lbl_p = torch.arange(len(u_vlogs), device=sim_p.device)
                loss_p = (F.cross_entropy(sim_p, lbl_p) + F.cross_entropy(sim_p.T, lbl_p)) / 2.0
            else:
                loss_p = loss_u
        else:
            loss_p = loss_u

        return (loss_u + loss_v + loss_p) / 3.0

    def denoising_consistency_loss(self, user_v, user_a, batch_users):
        """
        Auxiliary denoising consistency loss.
        """
        z_v = F.normalize(user_v[batch_users], dim=-1)
        z_a = F.normalize(user_a[batch_users], dim=-1)
        return F.mse_loss(z_v, z_a)

    # ────────────────────────────────────────────────────────────────
    # Edge reliability scoring  (Legacy CONDE denoising interface)
    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def compute_edge_reliability_uv(self, src_users, dst_videos):
        eu = self.emb_user.weight[src_users]
        ev = self.emb_video.weight[dst_videos]
        return self.edge_scorer_uv(eu, ev)

    @torch.no_grad()
    def compute_edge_reliability_ua(self, src_users, dst_vloggers):
        eu = self.emb_user.weight[src_users]
        ea = self.emb_vlogger.weight[dst_vloggers]
        return self.edge_scorer_ua(eu, ea)

    # ────────────────────────────────────────────────────────────────
    # getUsersRating – vectorised scoring with Bilinear Adaptive Gate
    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def getUsersRating(self, users, graph_uv, graph_ua, vlogger_list=None):
        """
        Returns a [|users|, n_videos] score matrix for evaluation.
        Uses exact VA-GNN Bilinear Adaptive Gate when enabled.
        """
        emb = self.forward(graph_uv, graph_ua)
        u_v = emb["user_v"][users]          # [B, D*(L+1)]
        v_v = emb["video_v"]                # [n_videos, D*(L+1)]
        u_a = emb["user_a"][users]          # [B, D*(L+1)]
        p_a = emb["vlogger_a"]              # [n_vloggers, D*(L+1)]

        s_video = torch.matmul(u_v, v_v.T)  # [B, n_videos]

        vl = vlogger_list if vlogger_list is not None else self.vlogger_list
        if vl is not None:
            if not isinstance(vl, torch.Tensor):
                vl = torch.LongTensor(vl).to(config.DEVICE)
            p_of_v = p_a[vl]                # [n_videos, D*(L+1)]
            s_vlogger = torch.matmul(u_a, p_of_v.T) # [B, n_videos]

            if getattr(config, "USE_ADAPTIVE_GATE", True):
                w = self.compute_adaptive_gate(v_v, p_of_v) # [n_videos]
                return s_video * w.unsqueeze(0) + s_vlogger * (1.0 - w.unsqueeze(0))
            else:
                return config.LAMBDA_VLOGGER * s_video + (1.0 - config.LAMBDA_VLOGGER) * s_vlogger

        return s_video
