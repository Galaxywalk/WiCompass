# -*- coding: utf-8 -*-
"""
Transformer helper modules for Point Transformer.
Includes: Residual, PreNorm, FeedForward, Attention, Transformer, TransformerBlock, TransitionDown, Backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import einsum
from .utils import index_points, square_distance


class Residual(nn.Module):
    """Residual connection wrapper."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """Pre-normalization wrapper using LayerNorm."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, kdim, qdim, vdim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        dim = qdim
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(kdim, inner_dim, bias=False)
        self.to_v = nn.Linear(vdim, inner_dim, bias=False)
        self.to_q = nn.Linear(qdim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k=None, v=None):
        if k is None or v is None:
            k, v = q, q
        b, n, _, h = *q.shape, self.heads
        
        kqv = self.to_k(k), self.to_q(q), self.to_v(v)
        k, q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kqv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """Standard Transformer encoder stack."""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(
                    kdim=dim, qdim=dim, vdim=dim, 
                    heads=heads, dim_head=dim_head, dropout=0.
                ))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class TransformerBlock(nn.Module):
    """Point Transformer block with local attention based on k-nearest neighbors."""
    def __init__(self, d_points, d_model, k):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    def forward(self, xyz, features):
        """
        Args:
            xyz: point coordinates, [B, N, 3]
            features: point features, [B, N, D]
        Returns:
            Updated features and attention weights
        """
        dists = square_distance(xyz, xyz)  # [B, N, N]
        knn_idx = dists.argsort()[:, :, :self.k]  # [B, N, K]
        knn_xyz = index_points(xyz, knn_idx)  # [B, N, K, 3]

        pre = features
        x = self.fc1(features)  # [B, N, d_model]
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q: [B, N, d_model], k: [B, N, K, d_model], v: [B, N, K, d_model]
        
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # [B, N, K, d_model]
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)  # [B, N, K, d_model]
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre  # [B, N, d_points]
        return res, attn


class TransitionDown(nn.Module):
    """Transition layer for feature transformation."""
    def __init__(self, in_channel, internal_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, internal_channel, 1),
            nn.BatchNorm1d(internal_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(internal_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, xyz, features):
        """
        Input:
            xyz: point position data, [B, N, 3]
            features: point feature data, [B, N, D]
        Return:
            new_features: transformed features, [B, N, out_channel]
        """
        new_features = torch.cat([xyz, features], dim=-1)  # [B, N, 3+D]
        new_features = new_features.permute(0, 2, 1)  # [B, 3+D, N]
        new_features = self.conv1(new_features)  # [B, internal_channel, N]
        new_features = self.conv2(new_features)  # [B, out_channel, N]
        new_features = new_features.permute(0, 2, 1)  # [B, N, out_channel]
        return new_features


class Backbone(nn.Module):
    """Point Transformer backbone with hierarchical feature extraction."""
    def __init__(self, nblocks, nneighbor, input_dim, transformer_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        append_blocks = nblocks - 1
        
        for i in range(append_blocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(channel // 2 + 3, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.append_blocks = append_blocks

    def forward(self, x):
        """
        Args:
            x: input point cloud, [B, N, C], C = (x, y, z, doppler, intensity, ...)
        Returns:
            points: extracted features
            xyz_and_feats: intermediate features at each level
        """
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]
        xyz_and_feats = [(xyz, points)]
        
        for i in range(self.append_blocks):
            points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        
        return points, xyz_and_feats

