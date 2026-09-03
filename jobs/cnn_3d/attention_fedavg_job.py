"""
Alias de compatibilité.
Le job principal s'appelle désormais AttentionCNN3DJob
(apprentissage distribué SGD, plus de mention FedAvg).
"""
from .attention_job import AttentionCNN3DJob as AttentionFedAvgCNN3DJob
from .attention_job import AttentionCNN3DJob

__all__ = ["AttentionCNN3DJob", "AttentionFedAvgCNN3DJob"]
