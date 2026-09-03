"""
Jobs progressifs (Phase 1 → 2 → 3) pour le framework de calcul volontaire
=========================================================================
Chaque job respecte l'interface TrainingJob :
  init_params, n_params, init_opt_state, make_task,
  compute_gradient, apply_gradient, evaluate
"""

from __future__ import annotations

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F

from framework.job import TrainingJob
from .models_pytorch import (
    CNN2D, CNN3D, CNN3DAttention,
    params_to_vector, vector_to_params, count_params,
)
from .data_providers import CIFAR10Provider, ModelNet40Provider


def _clip_grad_np(grad: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """Clip la norme L2 du gradient (stabilise SGD distribue)."""
    if max_norm is None or max_norm <= 0:
        return grad
    norm = float(np.linalg.norm(grad))
    if norm > max_norm and norm > 0:
        grad = grad * (max_norm / norm)
    return grad.astype(np.float32)


def _encode_batch(x: np.ndarray, y: np.ndarray) -> str:
    """Encode un mini-batch en float16 + npz compressé + base64 (serveur → volontaire)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, x=x.astype(np.float16), y=y.astype(np.int64))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_batch(payload: str):
    """Décode un mini-batch reçu dans une tâche."""
    raw = base64.b64decode(payload)
    npz = np.load(io.BytesIO(raw))
    return npz["x"].astype(np.float32), npz["y"].astype(np.int64)




def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  Phase 1 — CIFAR-10 / CNN 2D
# --------------------------------------------------------------------------- #
class Phase1CIFAR2DJob(TrainingJob):
    name = "Phase1-CIFAR10-CNN2D"

    def __init__(self, n_classes: int = 10, seed: int = 0, lr: float = 1e-3):
        self.n_classes = n_classes
        self.seed = seed
        self.lr = lr
        self.device = _get_device()
        self.model = CNN2D(n_classes=n_classes).to(self.device)
        self.data = CIFAR10Provider(n_classes=n_classes)
        torch.manual_seed(seed)

    def kb_spec(self):
        return {"phase": 1, "n_classes": self.n_classes, "kind": "cifar2d"}

    @classmethod
    def from_kb_spec(cls, spec, cfg):
        return cls(n_classes=spec.get("n_classes", 10))

    def init_params(self):
        return params_to_vector(self.model)

    def n_params(self):
        return count_params(self.model)

    def init_opt_state(self):
        n = self.n_params()
        return {"m": np.zeros(n, np.float32), "v": np.zeros(n, np.float32), "t": 0}

    def make_task(self, epoch, index, seed, epsilon):
        return {
            "epoch": epoch, "index": index, "seed": int(seed),
            "batch_size": 64, "epsilon": float(epsilon),
        }

    def enrich_task_with_data(self, task: dict) -> dict:
        """Serveur uniquement : échantillonne et encode le mini-batch."""
        rng = np.random.default_rng(task["seed"])
        batch_size = int(task.get("batch_size", 64))
        x, y = self.data.sample_batch(rng, batch_size)
        task = dict(task)
        task["batch_payload"] = _encode_batch(x, y)
        task["batch_shape"] = list(x.shape)
        task["n_samples"] = int(len(y))
        return task

    def compute_gradient(self, params_flat, task):
        vector_to_params(self.model, params_flat)
        self.model.train()

        if "batch_payload" in task:
            # Chemin principal : données fournies par le serveur (aucun dataset local)
            x_np, y_np = _decode_batch(task["batch_payload"])
        else:
            # Fallback tests / rétrocompatibilité
            rng = np.random.default_rng(task["seed"])
            x_np, y_np = self.data.sample_batch(rng, int(task.get("batch_size", 64)))

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).long().to(self.device)

        self.model.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grads = []
        for p in self.model.parameters():
            if p.grad is None:
                grads.append(np.zeros(p.numel(), np.float32))
            else:
                grads.append(p.grad.detach().cpu().numpy().ravel())
        grad = np.concatenate(grads).astype(np.float32)
        grad = _clip_grad_np(grad, max_norm=1.0)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()

        return grad, int(y.size(0)), {"loss": float(loss.item()), "accuracy": float(acc)}

    def apply_gradient(self, params, grad, opt_state, lr):
        # Adam simple cote serveur
        m, v, t = opt_state["m"], opt_state["v"], opt_state["t"]
        t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        params = params - lr * mhat / (np.sqrt(vhat) + eps)
        return params.astype(np.float32), {"m": m, "v": v, "t": t}

    def evaluate(self, params_flat, n=1000):
        vector_to_params(self.model, params_flat)
        self.model.eval()
        x_np, y_np = self.data.sample_eval(n)
        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).long().to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            loss = F.cross_entropy(logits, y).item()
            acc = (logits.argmax(1) == y).float().mean().item()
        return {"accuracy": float(acc), "loss": float(loss)}


# --------------------------------------------------------------------------- #
#  Phase 2 — ModelNet40 / CNN 3D
# --------------------------------------------------------------------------- #
class Phase2ModelNet3DJob(TrainingJob):
    name = "Phase2-ModelNet40-CNN3D"

    def __init__(self, n_classes: int = 40, seed: int = 0, lr: float = 1e-3):
        self.n_classes = n_classes
        self.seed = seed
        self.lr = lr
        self.device = _get_device()
        self.model = CNN3D(n_classes=n_classes).to(self.device)
        self.data = ModelNet40Provider(n_classes=n_classes)
        torch.manual_seed(seed)

    def kb_spec(self):
        return {"phase": 2, "n_classes": self.n_classes, "kind": "modelnet3d"}

    @classmethod
    def from_kb_spec(cls, spec, cfg):
        return cls(n_classes=spec.get("n_classes", 40))

    def init_params(self):
        return params_to_vector(self.model)

    def n_params(self):
        return count_params(self.model)

    def init_opt_state(self):
        n = self.n_params()
        return {"m": np.zeros(n, np.float32), "v": np.zeros(n, np.float32), "t": 0}

    def make_task(self, epoch, index, seed, epsilon):
        return {
            "epoch": epoch, "index": index, "seed": int(seed),
            "batch_size": 16, "epsilon": float(epsilon),
        }

    def enrich_task_with_data(self, task: dict) -> dict:
        """Serveur uniquement : échantillonne et encode le mini-batch (voxels déjà prêts côté serveur)."""
        rng = np.random.default_rng(task["seed"])
        batch_size = int(task.get("batch_size", 16))
        x, y = self.data.sample_batch(rng, batch_size)
        task = dict(task)
        task["batch_payload"] = _encode_batch(x, y)
        task["batch_shape"] = list(x.shape)
        task["n_samples"] = int(len(y))
        return task

    def compute_gradient(self, params_flat, task):
        vector_to_params(self.model, params_flat)
        self.model.train()

        if "batch_payload" in task:
            x_np, y_np = _decode_batch(task["batch_payload"])
        else:
            rng = np.random.default_rng(task["seed"])
            x_np, y_np = self.data.sample_batch(rng, int(task.get("batch_size", 16)))

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).long().to(self.device)

        self.model.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grads = [p.grad.detach().cpu().numpy().ravel() if p.grad is not None
                 else np.zeros(p.numel(), np.float32) for p in self.model.parameters()]
        grad = np.concatenate(grads).astype(np.float32)
        grad = _clip_grad_np(grad, max_norm=1.0)

        with torch.no_grad():
            acc = (logits.argmax(1) == y).float().mean().item()
        return grad, int(y.size(0)), {"loss": float(loss.item()), "accuracy": float(acc)}

    def apply_gradient(self, params, grad, opt_state, lr):
        m, v, t = opt_state["m"], opt_state["v"], opt_state["t"]
        t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        params = params - lr * mhat / (np.sqrt(vhat) + eps)
        return params.astype(np.float32), {"m": m, "v": v, "t": t}

    def evaluate(self, params_flat, n=800):
        vector_to_params(self.model, params_flat)
        self.model.eval()
        x_np, y_np = self.data.sample_eval(n)
        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).long().to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            loss = F.cross_entropy(logits, y).item()
            acc = (logits.argmax(1) == y).float().mean().item()
        return {"accuracy": float(acc), "loss": float(loss)}


# --------------------------------------------------------------------------- #
#  Phase 3 — ModelNet40 / CNN 3D + Attention
# --------------------------------------------------------------------------- #
class Phase3Attention3DJob(TrainingJob):
    name = "Phase3-ModelNet40-CNN3D-Attention"

    def __init__(self, n_classes: int = 40, seed: int = 0, lr: float = 1e-3):
        self.n_classes = n_classes
        self.seed = seed
        self.lr = lr
        self.device = _get_device()
        self.model = CNN3DAttention(n_classes=n_classes).to(self.device)
        self.data = ModelNet40Provider(n_classes=n_classes)
        torch.manual_seed(seed)

    def kb_spec(self):
        return {"phase": 3, "n_classes": self.n_classes, "kind": "attention3d"}

    @classmethod
    def from_kb_spec(cls, spec, cfg):
        return cls(n_classes=spec.get("n_classes", 40))

    def init_params(self):
        return params_to_vector(self.model)

    def n_params(self):
        return count_params(self.model)

    def init_opt_state(self):
        n = self.n_params()
        return {"m": np.zeros(n, np.float32), "v": np.zeros(n, np.float32), "t": 0}

    def make_task(self, epoch, index, seed, epsilon):
        return {
            "epoch": epoch, "index": index, "seed": int(seed),
            "batch_size": 16, "epsilon": float(epsilon),
        }

    def enrich_task_with_data(self, task: dict) -> dict:
        """Serveur uniquement : échantillonne et encode le mini-batch (voxels déjà prêts côté serveur)."""
        rng = np.random.default_rng(task["seed"])
        batch_size = int(task.get("batch_size", 16))
        x, y = self.data.sample_batch(rng, batch_size)
        task = dict(task)
        task["batch_payload"] = _encode_batch(x, y)
        task["batch_shape"] = list(x.shape)
        task["n_samples"] = int(len(y))
        return task

    def compute_gradient(self, params_flat, task):
        vector_to_params(self.model, params_flat)
        self.model.train()

        if "batch_payload" in task:
            x_np, y_np = _decode_batch(task["batch_payload"])
        else:
            rng = np.random.default_rng(task["seed"])
            x_np, y_np = self.data.sample_batch(rng, int(task.get("batch_size", 16)))

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).long().to(self.device)

        self.model.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grads = [p.grad.detach().cpu().numpy().ravel() if p.grad is not None
                 else np.zeros(p.numel(), np.float32) for p in self.model.parameters()]
        grad = np.concatenate(grads).astype(np.float32)
        grad = _clip_grad_np(grad, max_norm=1.0)

        with torch.no_grad():
            acc = (logits.argmax(1) == y).float().mean().item()
        return grad, int(y.size(0)), {"loss": float(loss.item()), "accuracy": float(acc)}

    def apply_gradient(self, params, grad, opt_state, lr):
        m, v, t = opt_state["m"], opt_state["v"], opt_state["t"]
        t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        params = params - lr * mhat / (np.sqrt(vhat) + eps)
        return params.astype(np.float32), {"m": m, "v": v, "t": t}

    def evaluate(self, params_flat, n=800):
        vector_to_params(self.model, params_flat)
        self.model.eval()
        x_np, y_np = self.data.sample_eval(n)
        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).long().to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            loss = F.cross_entropy(logits, y).item()
            acc = (logits.argmax(1) == y).float().mean().item()
        return {"accuracy": float(acc), "loss": float(loss)}
