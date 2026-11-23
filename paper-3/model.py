"""
model.py – MPGCL-Inspired Dual Meta-Path GNN with Graph Denoising and NT-Xent Contrastive Loss.
Implements Idea C (Paper 3 Meta-Path Contrastive Framework + Vynora CONDE Denoising).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ProjectionHead(nn.Module):
    """
    Non-linear projection head for contrastive learning (Paper 3 Equation 9).
    """
    def __init__(self, in_dim=config.EMBED_DIM, out_dim=config.PROJECTION_DIM, drop=config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)


class VynoraPaper3GNN(nn.Module):
    """
    Full Architecture combining:
    1. Bipartite GNN for User-Item collaborative filtering (with graph denoising).
    2. Meta-Path Homogeneous GNNs (U-I-U and I-U-I) for capturing structural heterogeneity.
    3. Type-specific NT-Xent contrastive learning across augmented homogeneous views.
    """
    def __init__(self, n_users, n_items, embed_dim=config.EMBED_DIM, n_layers=config.GNN_LAYERS):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        
        # 1. Base Node Embeddings (Gaussian 0, 0.02 matching Paper 3)
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        
        # 2. Homogeneous GCN Layers for Meta-Paths
        self.proj_user = ProjectionHead(embed_dim, embed_dim)
        self.proj_item = ProjectionHead(embed_dim, embed_dim)
        
        # 3. Graph Adjacency Placeholders
        self.graph_ui = None
        self.meta_graph_u = None
        self.meta_graph_i = None

    def set_graphs(self, graph_ui, meta_graph_u, meta_graph_i):
        self.graph_ui = graph_ui
        self.meta_graph_u = meta_graph_u
        self.meta_graph_i = meta_graph_i

    def _propagate_bipartite(self, graph_ui):
        """
        LightGCN propagation over bipartite interaction graph.
        """
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(graph_ui, all_emb)
            embs.append(all_emb)
        final_emb = torch.mean(torch.stack(embs, dim=0), dim=0)
        return final_emb[:self.n_users], final_emb[self.n_users:]

    def _propagate_homogeneous(self, adj, x):
        """
        GCN propagation over homogeneous meta-path graph (Paper 3 Equation 7).
        """
        out = x
        for _ in range(2):
            out = torch.sparse.mm(adj, out)
            out = F.leaky_relu(out, negative_slope=0.2)
        return out

    def get_bipartite_embeddings(self):
        """
        Returns current user and item embeddings from the bipartite graph.
        """
        return self._propagate_bipartite(self.graph_ui)

    def forward(self):
        """
        Forward pass producing final fused representations (h_u, h_v).
        """
        u_bip, i_bip = self._propagate_bipartite(self.graph_ui)
        
        # Meta-path representations
        u_meta = self._propagate_homogeneous(self.meta_graph_u, self.user_embedding.weight)
        i_meta = self._propagate_homogeneous(self.meta_graph_i, self.item_embedding.weight)
        
        # Fuse representations (Paper 3 Section 4)
        h_u = u_bip + 0.5 * u_meta
        h_v = i_bip + 0.5 * i_meta
        return h_u, h_v

    def compute_contrastive_loss(self, batch_u, batch_i):
        """
        Computes NT-Xent loss over augmented homogeneous views (Paper 3 Equation 8 & 10).
        """
        # Node embeddings
        u_emb = self.user_embedding(batch_u)
        i_emb = self.item_embedding(batch_i)
        
        # Augmented views via latent perturbation (GraphCL / MPGCL)
        noise_u1 = torch.randn_like(u_emb) * 0.05
        noise_u2 = torch.randn_like(u_emb) * 0.05
        u_z1 = F.normalize(self.proj_user(u_emb + noise_u1), dim=-1)
        u_z2 = F.normalize(self.proj_user(u_emb + noise_u2), dim=-1)
        
        noise_i1 = torch.randn_like(i_emb) * 0.05
        noise_i2 = torch.randn_like(i_emb) * 0.05
        i_z1 = F.normalize(self.proj_item(i_emb + noise_i1), dim=-1)
        i_z2 = F.normalize(self.proj_item(i_emb + noise_i2), dim=-1)
        
        # NT-Xent for Users
        sim_u_pos = torch.sum(u_z1 * u_z2, dim=-1) / config.TEMPERATURE
        sim_u_all = torch.mm(u_z1, u_z2.t()) / config.TEMPERATURE
        loss_u = -torch.mean(sim_u_pos - torch.logsumexp(sim_u_all, dim=-1))
        
        # NT-Xent for Items
        sim_i_pos = torch.sum(i_z1 * i_z2, dim=-1) / config.TEMPERATURE
        sim_i_all = torch.mm(i_z1, i_z2.t()) / config.TEMPERATURE
        loss_i = -torch.mean(sim_i_pos - torch.logsumexp(sim_i_all, dim=-1))
        
        return loss_u + loss_i

    def compute_bpr_loss(self, h_u, h_v, users, pos_items, neg_items):
        """
        Supervised Bayesian Personalized Ranking Loss (Paper 3 Equation 13).
        """
        u_vec = h_u[users]
        pos_vec = h_v[pos_items]
        neg_vec = h_v[neg_items]
        
        pos_scores = torch.sum(u_vec * pos_vec, dim=-1)
        neg_scores = torch.sum(u_vec * neg_vec, dim=-1)
        
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 Regularization
        reg_loss = config.WEIGHT_DECAY * (
            torch.norm(self.user_embedding(users)) ** 2 +
            torch.norm(self.item_embedding(pos_items)) ** 2 +
            torch.norm(self.item_embedding(neg_items)) ** 2
        ) / len(users)
        
        return bpr_loss, reg_loss
