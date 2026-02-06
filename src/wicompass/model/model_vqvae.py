#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint VQ-VAE Model inspired by PCT architecture
Joint VQ-VAE model based on PCT architecture, maintaining PCT structural characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Tuple, Optional
import math


class MixerLayer(nn.Module):
    """MLP-Mixer layer following PCT implementation"""
    def __init__(self, hidden_dim: int, hidden_inter_dim: int, 
                 token_num: int, token_inter_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_num = token_num
        
        # Token mixing (spatial) - following PCT
        self.token_mixer = nn.Sequential(
            nn.Linear(token_num, token_inter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_inter_dim, token_num),
            nn.Dropout(dropout)
        )
        
        # Channel mixing (feature) - following PCT
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_inter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_inter_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (B, token_num, hidden_dim) - consistent with PCT
        # Token mixing - following PCT residual connection
        x = x + self.token_mixer(x.transpose(1, 2)).transpose(1, 2)
        # Channel mixing - following PCT residual connection
        x = x + self.channel_mixer(x)
        return x


class JointTokenizer(nn.Module):
    """
    Joint Tokenizer based on PCT tokenizer architecture
    Contains encoder, codebook, and decoder
    """
    def __init__(self,
                 # Basic parameters
                 num_joints: int = 22,
                 input_dim: int = 3,  # Dimension per joint (x, y, z)
                 
                 # Encoder parameters
                 enc_num_blocks: int = 4,
                 enc_hidden_dim: int = 256,
                 enc_token_inter_dim: int = 512,  # Intermediate dimension for token mixing
                 enc_hidden_inter_dim: int = 1024,  # Intermediate dimension for channel mixing
                 enc_dropout: float = 0.1,
                 
                 # Codebook parameters
                 token_num: int = 64,  # Number of tokens per sample
                 token_class_num: int = 512,  # Codebook size
                 token_dim: int = 256,  # Token dimension
                 ema_decay: float = 0.99,
                 
                 # Decoder parameters
                 dec_num_blocks: int = 4,
                 dec_hidden_dim: int = 256,
                 dec_token_inter_dim: int = 512,
                 dec_hidden_inter_dim: int = 1024,
                 dec_dropout: float = 0.1,
                 
                 # Training parameters
                 drop_rate: float = 0.0):  # random masking rate
        
        super().__init__()
        
        # Store all configuration parameters
        self.num_joints = num_joints
        self.input_dim = input_dim
        
        # Encoder configuration
        self.enc_num_blocks = enc_num_blocks
        self.enc_hidden_dim = enc_hidden_dim
        self.enc_token_inter_dim = enc_token_inter_dim
        self.enc_hidden_inter_dim = enc_hidden_inter_dim
        self.enc_dropout = enc_dropout
        
        # Codebook configuration
        self.token_num = token_num
        self.token_class_num = token_class_num
        self.token_dim = token_dim
        self.ema_decay = ema_decay
        
        # Decoder configuration
        self.dec_num_blocks = dec_num_blocks
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_token_inter_dim = dec_token_inter_dim
        self.dec_hidden_inter_dim = dec_hidden_inter_dim
        self.dec_dropout = dec_dropout
        
        # Training configuration
        self.drop_rate = drop_rate
        
        # === Encoder section ===
        # Input embedding - map 3D joint coordinates to hidden dimension
        self.start_embed = nn.Linear(input_dim, enc_hidden_dim)
        
        # Invisible token (for masking, similar to PCT)
        self.invisible_token = nn.Parameter(torch.zeros(1, 1, enc_hidden_dim))
        nn.init.trunc_normal_(self.invisible_token, mean=0., std=0.02, a=-0.02, b=0.02)
        
        # Encoder Mixer layers
        self.encoder = nn.ModuleList([
            MixerLayer(enc_hidden_dim, enc_hidden_inter_dim, 
                      num_joints, enc_token_inter_dim, enc_dropout) 
            for _ in range(enc_num_blocks)
        ])
        self.encoder_layer_norm = nn.LayerNorm(enc_hidden_dim)
        
        # Token generation layer
        self.token_mlp = nn.Linear(num_joints, token_num)
        self.feature_embed = nn.Linear(enc_hidden_dim, token_dim)
        
        # === Codebook section (core of VQ-VAE) ===
        # Register codebook-related buffers (not updated by gradients)
        self.register_buffer('codebook', torch.empty(token_class_num, token_dim))
        self.codebook.data.normal_()
        
        # Buffers for EMA updates
        self.register_buffer('ema_cluster_size', torch.zeros(token_class_num))
        self.register_buffer('ema_w', torch.empty(token_class_num, token_dim))
        self.ema_w.data.normal_()
        
        # === Decoder section ===
        # Token recovery layer
        self.decoder_token_mlp = nn.Linear(token_num, num_joints)
        self.decoder_start = nn.Linear(token_dim, dec_hidden_dim)
        
        # Decoder Mixer layers
        self.decoder = nn.ModuleList([
            MixerLayer(dec_hidden_dim, dec_hidden_inter_dim,
                      num_joints, dec_token_inter_dim, dec_dropout) 
            for _ in range(dec_num_blocks)
        ])
        self.decoder_layer_norm = nn.LayerNorm(dec_hidden_dim)
        
        # Output layer - restore to original joint dimension
        self.recover_embed = nn.Linear(dec_hidden_dim, input_dim)
    
    def encode(self, joints, joints_visible=None, train=True):
        """
        Encoding stage: encode joint positions into discrete tokens
        
        Args:
            joints: (B, num_joints, input_dim) joint positions
            joints_visible: (B, num_joints) joint visibility mask
            train: whether in training mode
            
        Returns:
            encoding_indices: quantized token indices
            quantized_features: quantized features
            e_latent_loss: VQ loss
        """
        B = joints.shape[0]
        
        # Input embedding
        encode_feat = self.start_embed(joints)  # (B, num_joints, enc_hidden_dim)
        
        # Random masking during training (like PCT)
        if train and joints_visible is not None and self.drop_rate > 0:
            rand_mask_ind = torch.rand(joints_visible.shape, device=joints.device) > self.drop_rate
            joints_visible = torch.logical_and(rand_mask_ind, joints_visible)
        
        # Apply visibility mask
        if joints_visible is not None:
            mask_tokens = self.invisible_token.expand(B, self.num_joints, -1)
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens)
            encode_feat = encode_feat * w + mask_tokens * (1 - w)
        
        # Pass through encoder
        for layer in self.encoder:
            encode_feat = layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        
        # Generate tokens
        encode_feat = encode_feat.transpose(2, 1)  # (B, enc_hidden_dim, num_joints)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)  # (B, token_num, enc_hidden_dim)
        encode_feat = self.feature_embed(encode_feat)  # (B, token_num, token_dim)
        encode_feat_flat = encode_feat.flatten(0, 1)  # (B*token_num, token_dim)
        
        # Vector Quantization
        distances = torch.sum(encode_feat_flat**2, dim=1, keepdim=True) \
                    + torch.sum(self.codebook**2, dim=1) \
                    - 2 * torch.matmul(encode_feat_flat, self.codebook.t())
        
        encoding_indices = torch.argmin(distances, dim=1)  # (B*token_num,)
        encodings = torch.zeros(encoding_indices.shape[0], self.token_class_num, 
                               device=joints.device, dtype=encode_feat.dtype)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Get quantized features
        quantized_features = torch.matmul(encodings, self.codebook)  # (B*token_num, token_dim)
        quantized_features = quantized_features.view(B, self.token_num, self.token_dim)
        
        # VQ loss and straight-through estimator
        if train:
            # Update codebook with EMA
            if dist.is_available() and dist.is_initialized():
                # Synchronize in distributed training
                dw = torch.matmul(encodings.t(), encode_feat_flat.detach())
                n_encodings, n_dw = encodings.numel(), dw.numel()
                encodings_shape, dw_shape = encodings.shape, dw.shape
                combined = torch.cat((encodings.flatten(), dw.flatten()))
                dist.all_reduce(combined)
                sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
                sync_encodings = sync_encodings.view(encodings_shape)
                sync_dw = sync_dw.view(dw_shape)
            else:
                sync_encodings = encodings
                sync_dw = torch.matmul(encodings.t(), encode_feat_flat.detach())
            
            # Update EMA parameters
            self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                torch.sum(sync_encodings, 0), alpha=1 - self.ema_decay)
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size.data.add_(1e-5).div_(
                n + self.token_class_num * 1e-5).mul_(n)
            
            self.ema_w.data.mul_(self.ema_decay).add_(sync_dw, alpha=1 - self.ema_decay)
            self.codebook.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
            
            # VQ loss (Commitment Loss) - make encoder output close to quantization result
            e_latent_loss = F.mse_loss(quantized_features.detach(), encode_feat)
            
            # Straight-through estimator - use quantized result in forward pass, encoder result in backward pass
            quantized_features = encode_feat + (quantized_features - encode_feat).detach()
        else:
            e_latent_loss = None
        
        return encoding_indices.view(B, self.token_num), quantized_features, e_latent_loss
    
    def decode(self, quantized_features):
        """
        Decoding stage: decode quantized features into joint positions
        
        Args:
            quantized_features: (B, token_num, token_dim) quantized features
            
        Returns:
            recovered_joints: (B, num_joints, input_dim) recovered joint positions
        """
        B = quantized_features.shape[0]
        
        # Token dimension transformation
        part_token_feat = quantized_features.transpose(2, 1)  # (B, token_dim, token_num)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2, 1)  # (B, num_joints, token_dim)
        
        # Decoder input projection
        decode_feat = self.decoder_start(part_token_feat)  # (B, num_joints, dec_hidden_dim)
        
        # Pass through decoder
        for layer in self.decoder:
            decode_feat = layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)
        
        # Output projection
        recovered_joints = self.recover_embed(decode_feat)  # (B, num_joints, input_dim)
        
        return recovered_joints
    
    def forward(self, joints, joints_visible=None, cls_logits=None, train=True):
        """
        Forward pass
        
        Args:
            joints: (B, num_joints, input_dim) joint positions
            joints_visible: (B, num_joints) joint visibility (optional)
            cls_logits: classifier output (for classifier stage)
            train: whether in training mode
            
        Returns:
            recovered_joints: reconstructed joint positions
            encoding_indices: encoding indices
            e_latent_loss: VQ loss
        """
        if cls_logits is not None:
            # Classifier stage: directly use classifier output
            B = cls_logits.shape[0] // self.token_num
            quantized_features = torch.matmul(cls_logits, self.codebook)
            quantized_features = quantized_features.view(B, self.token_num, self.token_dim)
            encoding_indices = None
            e_latent_loss = None
        else:
            # Tokenizer stage: encode-quantize-decode
            encoding_indices, quantized_features, e_latent_loss = self.encode(
                joints, joints_visible, train)
        
        # Decode
        recovered_joints = self.decode(quantized_features)
        
        return recovered_joints, encoding_indices, e_latent_loss


class JointVQVAELoss(nn.Module):
    """Joint VQ-VAE loss function"""
    def __init__(self, 
                 recon_weight: float = 1.0,
                 vq_weight: float = 1.0,
                 commitment_weight: float = 0.25,
                 recon_loss_type: str = 'mse'):
        super().__init__()
        self.recon_weight = recon_weight
        self.vq_weight = vq_weight
        self.commitment_weight = commitment_weight
        
        if recon_loss_type == 'mse':
            self.recon_loss_fn = nn.MSELoss()
        elif recon_loss_type == 'l1':
            self.recon_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown reconstruction loss type: {recon_loss_type}")
    
    def forward(self, recovered_joints, target_joints, e_latent_loss=None):
        """
        Compute VQ-VAE loss
        
        Args:
            recovered_joints: (B, num_joints, 3) reconstructed joint positions
            target_joints: (B, num_joints, 3) target joint positions
            e_latent_loss: VQ loss (during training)
            
        Returns:
            loss_dict: dictionary containing all loss components
        """
        # Reconstruction loss
        recon_loss = self.recon_loss_fn(recovered_joints, target_joints)
        
        # Ensure loss is scalar to avoid DataParallel warnings
        if recon_loss.dim() > 0:
            recon_loss = recon_loss.mean()
        
        losses = {
            'recon_loss': recon_loss,
            'total_loss': self.recon_weight * recon_loss
        }
        
        # VQ loss (only during training) - this is commitment loss
        if e_latent_loss is not None:
            # Ensure VQ loss is also scalar
            if hasattr(e_latent_loss, 'dim') and e_latent_loss.dim() > 0:
                e_latent_loss = e_latent_loss.mean()
            
            # Apply commitment weight
            commitment_loss = self.commitment_weight * e_latent_loss
            losses['vq_loss'] = e_latent_loss  # Original commitment loss for monitoring
            losses['commitment_loss'] = e_latent_loss  # Add commitment_loss field for training code compatibility
            losses['total_loss'] = losses['total_loss'] + commitment_loss
        
        return losses


def create_joint_tokenizer(config: Dict) -> JointTokenizer:
    """
    Create Joint Tokenizer model from configuration
    
    Args:
        config: model configuration dictionary containing all necessary parameters
        
    Returns:
        JointTokenizer model instance
    """
    return JointTokenizer(
        # Basic parameters
        num_joints=config.get('num_joints', 22),
        input_dim=config.get('input_dim', 3),
        
        # Encoder parameters
        enc_num_blocks=config.get('enc_num_blocks', 4),
        enc_hidden_dim=config.get('enc_hidden_dim', 256),
        enc_token_inter_dim=config.get('enc_token_inter_dim', 512),
        enc_hidden_inter_dim=config.get('enc_hidden_inter_dim', 1024),
        enc_dropout=config.get('enc_dropout', 0.1),
        
        # Codebook parameters
        token_num=config.get('token_num', 64),
        token_class_num=config.get('token_class_num', 512),
        token_dim=config.get('token_dim', 256),
        ema_decay=config.get('ema_decay', 0.99),
        
        # Decoder parameters
        dec_num_blocks=config.get('dec_num_blocks', 4),
        dec_hidden_dim=config.get('dec_hidden_dim', 256),
        dec_token_inter_dim=config.get('dec_token_inter_dim', 512),
        dec_hidden_inter_dim=config.get('dec_hidden_inter_dim', 1024),
        dec_dropout=config.get('dec_dropout', 0.1),
        
        # Training parameters
        drop_rate=config.get('drop_rate', 0.0)
    )


# Usage example
if __name__ == "__main__":
    # Test model
    config = {
        'num_joints': 22,
        'input_dim': 3,
        
        # Encoder configuration
        'enc_num_blocks': 4,
        'enc_hidden_dim': 256,
        'enc_token_inter_dim': 512,
        'enc_hidden_inter_dim': 1024,
        'enc_dropout': 0.1,
        
        # Codebook configuration
        'token_num': 64,
        'token_class_num': 512,
        'token_dim': 256,
        'ema_decay': 0.99,
        
        # Decoder configuration
        'dec_num_blocks': 4,
        'dec_hidden_dim': 256,
        'dec_token_inter_dim': 512,
        'dec_hidden_inter_dim': 1024,
        'dec_dropout': 0.1,
        
        # Training configuration
        'drop_rate': 0.1
    }
    
    model = create_joint_tokenizer(config)
    
    # Test input
    batch_size = 8
    joints = torch.randn(batch_size, 22, 3)
    joints_visible = torch.ones(batch_size, 22).bool()
    
    # Forward pass
    recovered_joints, encoding_indices, e_latent_loss = model(
        joints, joints_visible, train=True)
    
    print(f"Input shape: {joints.shape}")
    print(f"Reconstructed shape: {recovered_joints.shape}")
    print(f"Encoding indices shape: {encoding_indices.shape if encoding_indices is not None else None}")
    print(f"VQ loss: {e_latent_loss.item() if e_latent_loss is not None else None}")
    
    # Test loss
    criterion = JointVQVAELoss()
    loss_dict = criterion(recovered_joints, joints, e_latent_loss)
    print(f"Loss dictionary: {loss_dict}")
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test all configuration parameters
    print("\nModel configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
