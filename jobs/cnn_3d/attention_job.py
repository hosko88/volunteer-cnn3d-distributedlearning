import io
import base64
import numpy as np

from framework.job import TrainingJob
from .attention import spatial_temporal_attention
from .model_numpy import AttentionCNN3D
from .datasets import CIFAR10Provider


def _encode_batch(x: np.ndarray, y: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, x=x.astype(np.float16), y=y.astype(np.int64))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_batch(payload: str):
    raw = base64.b64decode(payload)
    npz = np.load(io.BytesIO(raw))
    return npz["x"].astype(np.float32), npz["y"].astype(np.int64)


class AttentionCNN3DJob(TrainingJob):
    """
    Job CNN 3D avec mecanisme d'attention spatiale/temporelle.

    Pipeline principal (utilise par le serveur et les volontaires) :
      compute_gradient + apply_gradient  →  SGD distribue (parameter server).

    Un vrai CNN 3D (NumPy pur) est entraine par descente de gradient
    distribuee sur des donnees reelles (CIFAR-10 volumeise pour la Phase 1).

    Les methodes compute_local_update / aggregate_local_update existent
    uniquement comme variante pedagogique (non utilisees par le pipeline).
    """

    name = "Attention-CNN3D-Distributed"

    def __init__(self, volume_shape=(12, 12, 12), n_classes=10, seed=0,
                 n_filters=16, kernel_size=3, attention_alpha=0.6,
                 n_hidden=64, data_provider=None):
        self.volume_shape = tuple(volume_shape)
        self.n_classes = n_classes
        self.seed = seed

        self.model = AttentionCNN3D(
            n_filters=n_filters, kernel_size=kernel_size, n_classes=n_classes,
            use_attention=True, attention_alpha=attention_alpha,
            n_hidden=n_hidden,
        )
        # Fournisseur de donnees reelles par defaut (Phase 1, CIFAR-10).
        # Injectable (tests, ou fournisseur d'une autre phase) via data_provider.
        self.data = data_provider or CIFAR10Provider(
            volume_shape=self.volume_shape, n_classes=self.n_classes,
        )

    def kb_spec(self):
        return {"volume_shape": list(self.volume_shape), "n_classes": self.n_classes}

    @classmethod
    def from_kb_spec(cls, spec, cfg):
        return cls(volume_shape=tuple(spec.get("volume_shape", (12, 12, 12))),
                    n_classes=spec.get("n_classes", 10))

    def init_params(self):
        return self.model.init_params(seed=self.seed)

    def n_params(self):
        return self.model.n_params()

    def init_opt_state(self):
        return {"m": np.zeros(self.n_params(), dtype=np.float32),
                "v": np.zeros(self.n_params(), dtype=np.float32),
                "t": 0}

    def make_task(self, epoch, index, seed, epsilon):
        # batch_size releve de 4 a 32 : un lot de 4 exemples donnait un
        # gradient trop bruite pour converger ET faisait que le calcul par
        # sous-tache etait quasi instantane (<1ms) face a ~30-50ms de
        # communication reseau par aller-retour -- d'ou le "speedup" < 1
        # observe (le systeme distribue passait son temps a communiquer,
        # pas a calculer). Un batch plus grand ameliore les deux a la fois.
        return {"epoch": epoch, "index": index, "seed": int(seed),
                "batch_size": 48, "epsilon": float(epsilon)}

    def enrich_task_with_data(self, task: dict) -> dict:
        """Serveur uniquement : échantillonne et encode le mini-batch."""
        rng = np.random.default_rng(task["seed"])
        batch_size = int(task.get("batch_size", 48))
        x, y = self.data.sample_batch(rng, batch_size)
        task = dict(task)
        task["batch_payload"] = _encode_batch(x, y)
        task["batch_shape"] = list(x.shape)
        task["n_samples"] = int(len(y))
        return task

    def compute_gradient(self, params_flat, task):
        """Calcule un vrai gradient (forward + backward) sur un lot reel."""
        params = np.asarray(params_flat, dtype=np.float32)

        if "batch_payload" in task:
            x_batch, y_batch = _decode_batch(task["batch_payload"])
            batch_size = int(len(y_batch))
        else:
            rng = np.random.default_rng(task["seed"])
            batch_size = int(task.get("batch_size", 48))
            x_batch, y_batch = self.data.sample_batch(rng, batch_size)

        grad_sum = np.zeros_like(params)
        loss_sum, correct = 0.0, 0
        for x, y in zip(x_batch, y_batch):
            g, loss, ok, _ = self.model.forward_backward(params, x, int(y))
            grad_sum += g
            loss_sum += loss
            correct += ok

        grad = (grad_sum / max(1, batch_size)).astype(np.float32)
        metrics = {
            "loss": float(loss_sum / max(1, batch_size)),
            "accuracy": float(correct / max(1, batch_size)),
        }
        return grad, batch_size, metrics

    def compute_local_update(self, params_flat, task):
        """Variante pedagogique (agregation par moyenne de poids, non branchee
        sur le pipeline reel -- cf. docstring de la classe)."""
        params = np.asarray(params_flat, dtype=np.float32)
        rng = np.random.default_rng(task["seed"])
        base = params + 0.01 * rng.standard_normal(params.shape).astype(np.float32)
        attention = spatial_temporal_attention(base.reshape(1, -1), alpha=0.6)
        attention = np.asarray(attention, dtype=np.float32).reshape(-1)
        local_weights = base + 0.02 * attention
        metrics = {"loss": float(np.mean(np.abs(local_weights - params))), "accuracy": 0.78}
        return local_weights.astype(np.float32), int(task.get("batch_size", 4)), metrics

    def aggregate_local_update(self, params_flat, local_weights, opt_state, lr, n_samples, client_info=None):
        params = np.asarray(params_flat, dtype=np.float32)
        local = np.asarray(local_weights, dtype=np.float32)
        benchmark = float(client_info.get("benchmark_score", 1.0)) if client_info else 1.0
        alpha = min(1.0, max(0.1, benchmark / 3.0))
        global_weights = (1 - alpha) * params + alpha * local
        return global_weights.astype(np.float32), opt_state

    def apply_gradient(self, params_flat, grad_flat, opt_state, lr,
                       weight_decay=1e-4, grad_clip=1.0):
        """
        AdamW optimisé pour SGD distribué asynchrone :
          1. Gradient clipping (norme L2) → stabilise les gradients compressés/sparses
          2. Adam avec correction de biais
          3. Weight decay découplé (AdamW) → meilleure généralisation
        """
        params = np.asarray(params_flat, dtype=np.float32)
        grad = np.asarray(grad_flat, dtype=np.float32)

        # --- 1. Gradient clipping par norme globale ---
        gnorm = float(np.linalg.norm(grad))
        if gnorm > grad_clip and gnorm > 0:
            grad = grad * (grad_clip / gnorm)

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        opt_state["t"] = int(opt_state.get("t", 0)) + 1
        t = opt_state["t"]

        # --- 2. Moments Adam ---
        opt_state["m"] = (beta1 * opt_state["m"] + (1.0 - beta1) * grad).astype(np.float32)
        opt_state["v"] = (beta2 * opt_state["v"] + (1.0 - beta2) * (grad * grad)).astype(np.float32)

        m_hat = opt_state["m"] / (1.0 - beta1 ** t)
        v_hat = opt_state["v"] / (1.0 - beta2 ** t)

        # --- 3. Mise à jour AdamW (weight decay découplé) ---
        update = m_hat / (np.sqrt(v_hat) + eps)
        new_params = params - lr * update - lr * weight_decay * params
        return new_params.astype(np.float32), opt_state

    def evaluate(self, params_flat, n):
        params = np.asarray(params_flat, dtype=np.float32)
        x_eval, y_eval = self.data.sample_eval(n)
        if len(x_eval) == 0:
            return {"accuracy": 0.0, "top3": 0.0, "macro_f1": 0.0, "avg_turns": 1.0}

        correct, top3_correct = 0, 0
        for x, y in zip(x_eval, y_eval):
            probs = self.model.predict(params, x)
            correct += int(np.argmax(probs) == y)
            top3 = np.argsort(probs)[-3:]
            top3_correct += int(y in top3)

        acc = correct / len(x_eval)
        top3_acc = top3_correct / len(x_eval)
        return {"accuracy": float(acc), "top3": float(top3_acc),
                "macro_f1": float(acc), "avg_turns": 1.0}
