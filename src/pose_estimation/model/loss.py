# -*- coding: utf-8 -*-
"""
Loss functions for human pose estimation.

MPJPE: Mean Per-Joint Position Error (Protocol #1)
P-MPJPE: Procrustes-aligned MPJPE (Protocol #2)
"""

import torch
import numpy as np


def mpjpe(predicted, target):
    """
    Mean Per-Joint Position Error (Protocol #1).
    
    Computes the mean Euclidean distance between predicted and ground truth joint positions.
    
    Args:
        predicted: Predicted joint positions [B, J, 3]
        target: Ground truth joint positions [B, J, 3]
    
    Returns:
        Mean error across all joints and samples (in same units as input)
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe(predicted, target):
    """
    Procrustes-aligned Mean Per-Joint Position Error (Protocol #2).
    
    Computes MPJPE after rigid alignment (scale, rotation, and translation)
    between predicted and ground truth poses.
    
    Args:
        predicted: Predicted joint positions [B, J, 3]
        target: Ground truth joint positions [B, J, 3]
    
    Returns:
        Mean error after Procrustes alignment (in same units as input)
    """
    assert predicted.shape == target.shape
    
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()

    # Compute mean positions
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    # Center the data
    X0 = target - muX
    Y0 = predicted - muY

    # Normalize
    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    # Compute optimal rotation using SVD
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e., rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    # Compute scale and translation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Apply rigid transformation to predicted poses
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE after alignment
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def weighted_mpjpe(predicted, target, weights):
    """
    Weighted Mean Per-Joint Position Error.
    
    Args:
        predicted: Predicted joint positions [B, J, 3]
        target: Ground truth joint positions [B, J, 3]
        weights: Per-joint weights [J] or [B, J]
    
    Returns:
        Weighted mean error
    """
    assert predicted.shape == target.shape
    per_joint_error = torch.norm(predicted - target, dim=-1)  # [B, J]
    return torch.mean(per_joint_error * weights)

