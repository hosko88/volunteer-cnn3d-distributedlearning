"""
Configuration centrale
=======================
Theme : Using Volunteer Computing for Distributed Learning with Convolutional
Neural Networks: A Progressive Approach Toward 3D Imaging
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    seed: int = 42


@dataclass
class ModelConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0          # norme max du gradient (clipping)
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.95


@dataclass
class TransportConfig:
    """
    Compression des gradients pour modeles ~0.5M–1.2M parametres.
    - fp16 dense : stable, x2
    - fp16 + topk 0.15 : fort gain bande passante
    - int8 + topk : encore plus compact, legerement plus bruite
    """
    dtype: str = "fp16"             # fp16 recommande pour gros modeles
    topk: float = 0.15              # 15% des plus grands |g| (bon compromis)
    error_feedback: bool = True     # stabilise la sparsification


@dataclass
class TrainConfig:
    n_tasks_per_epoch: int = 256
    batch_size: int = 32
    local_steps: int = 1            # accumulation locale (1 = classique)
    max_epochs: int = 250
    target_accuracy: float = 0.75
    completion_fraction: float = 0.75
    task_timeout: float = 180.0
    staleness_max: int = 40
    n_shards: int = 20               # DEPRECATED : plus utilisé.
                                     # Le serveur envoie désormais uniquement
                                     # les mini-batches nécessaires dans chaque tâche.


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    validation_samples: int = 1000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


DEFAULT = Config()
