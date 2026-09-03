"""
Modeles PyTorch reels pour la progression 2D -> 3D
==================================================
Phase 1 : CNN 2D classique (CIFAR-10)
Phase 2 : CNN 3D (ModelNet40 voxels)
Phase 3 : CNN 3D + Attention spatiale
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Phase 1 — CNN 2D (CIFAR-10)
# --------------------------------------------------------------------------- #
class CNN2D(nn.Module):
    """CNN 2D performant et raisonnable pour CIFAR-10."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Phase 2 — CNN 3D (ModelNet40)
# --------------------------------------------------------------------------- #
class CNN3D(nn.Module):
    """CNN 3D pour volumes voxelises (ModelNet40)."""

    def __init__(self, n_classes: int = 40, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x : (B, 1, D, H, W)
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Phase 3 — CNN 3D + Attention
# --------------------------------------------------------------------------- #
class SpatialAttention3D(nn.Module):
    """Attention spatiale simple sur les cartes 3D."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : (B, C, D, H, W)
        attn = self.sigmoid(self.conv(x))
        return x * attn


class CNN3DAttention(nn.Module):
    """CNN 3D avec module d'attention spatiale (Phase 3)."""

    def __init__(self, n_classes: int = 40, in_channels: int = 1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.attn1 = SpatialAttention3D(32)

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.attn2 = SpatialAttention3D(64)

        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.attn3 = SpatialAttention3D(128)

        self.layer4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.attn1(x)
        x = self.layer2(x)
        x = self.attn2(x)
        x = self.layer3(x)
        x = self.attn3(x)
        x = self.layer4(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Utilitaires : conversion parametres <-> vecteur plat (pour le framework)
# --------------------------------------------------------------------------- #
def params_to_vector(model: nn.Module) -> np.ndarray:
    """Serialise tous les parametres du modele en un vecteur float32."""
    vecs = [p.detach().cpu().numpy().ravel() for p in model.parameters()]
    return np.concatenate(vecs).astype(np.float32)


def vector_to_params(model: nn.Module, vec: np.ndarray) -> None:
    """Charge un vecteur plat dans les parametres du modele (in-place)."""
    vec = np.asarray(vec, dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = vec[offset:offset + n].reshape(p.shape)
            p.copy_(torch.from_numpy(chunk))
            offset += n


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
