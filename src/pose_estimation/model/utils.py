# -*- coding: utf-8 -*-
"""
Utility functions for Point Transformer model.
"""

import torch


def index_points(points, idx):
    """
    Index points from point cloud based on given indices.
    
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points: indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each pair of points.
    
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1) + sum(dst**2,dim=-1) - 2*src^T*dst
    
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

