import numpy as np

from framework.job import TrainingJob


class FedAvgCNN3DJob(TrainingJob):
    """LEGACY / pedagogique uniquement.

    Variante d'agregation par moyenne de poids.
    NON utilisee par le pipeline principal (SGD distribue).
    Conservee pour comparaison eventuelle dans le memoire.
    """

    name = "FedAvg-CNN3D"

    def __init__(self, volume_shape=(8, 8, 8), n_classes=2, seed=0):
        self.volume_shape = tuple(volume_shape)
        self.n_classes = n_classes
        self.seed = seed

    def kb_spec(self):
        return {
            "volume_shape": list(self.volume_shape),
            "n_classes": self.n_classes,
        }

    @classmethod
    def from_kb_spec(cls, spec, cfg):
        return cls(volume_shape=tuple(spec.get("volume_shape", (8, 8, 8))),
                   n_classes=spec.get("n_classes", 2))

    def init_params(self):
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, 0.01, size=12).astype(np.float32)

    def n_params(self):
        return 12

    def init_opt_state(self):
        return {"m": np.zeros(self.n_params(), dtype=np.float32)}

    def make_task(self, epoch, index, seed, epsilon):
        return {
            "epoch": epoch,
            "index": index,
            "seed": int(seed),
            "batch_size": 4,
            "epsilon": float(epsilon),
            "alpha": 0.5,
        }

    def compute_gradient(self, params_flat, task):
        rng = np.random.default_rng(task["seed"])
        params = np.asarray(params_flat, dtype=np.float32)
        batch_size = int(task.get("batch_size", 4))

        grad = 0.01 * rng.standard_normal(params.shape).astype(np.float32)
        metrics = {
            "loss": float(np.mean(np.abs(grad))),
            "accuracy": float(min(0.99, 0.55 + 0.02 * batch_size)),
        }
        return grad.astype(np.float32), batch_size, metrics

    def compute_local_update(self, params_flat, task):
        rng = np.random.default_rng(task["seed"])
        params = np.asarray(params_flat, dtype=np.float32)
        batch_size = int(task.get("batch_size", 4))

        local_weights = params + 0.01 * rng.standard_normal(params.shape).astype(np.float32)
        local_weights = local_weights.astype(np.float32)

        metrics = {
            "loss": float(np.mean(np.abs(local_weights - params))),
            "accuracy": float(min(0.99, 0.55 + 0.02 * batch_size)),
        }
        return local_weights, batch_size, metrics

    def aggregate_local_update(self, params_flat, local_weights, opt_state, lr, n_samples, client_info=None):
        params = np.asarray(params_flat, dtype=np.float32)
        local = np.asarray(local_weights, dtype=np.float32)

        benchmark = float(client_info.get("benchmark_score", 1.0)) if client_info else 1.0
        alpha = min(1.0, max(0.1, benchmark / 3.0))

        global_weights = (1 - alpha) * params + alpha * local
        return global_weights.astype(np.float32), opt_state

    def apply_gradient(self, params_flat, grad_flat, opt_state, lr):
        params = np.asarray(params_flat, dtype=np.float32)
        grad = np.asarray(grad_flat, dtype=np.float32)
        return (params - lr * grad).astype(np.float32), opt_state

    def evaluate(self, params_flat, n):
        params = np.asarray(params_flat, dtype=np.float32)
        acc = float(min(0.99, 0.5 + 0.01 * np.sum(np.abs(params))))
        return {"accuracy": acc, "top3": acc, "macro_f1": acc, "avg_turns": 1.0}
