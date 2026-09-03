"""
Modele CNN 3D (NumPy pur) avec attention — version renforcee
=============================================================
Architecture plus expressive pour de meilleures performances,
tout en restant executable sur des machines volontaires (pas de GPU requis) :

    Volume (D,H,W)
      -> Conv3D-1 (F filtres) + ReLU
      -> Conv3D-2 (2F filtres) + ReLU
      -> Attention spatiale/temporelle
      -> Pooling regional 2x2x2
      -> FC1 (hidden) + ReLU
      -> FC2 (classes) + Softmax

Ameliorations par rapport a la version legere :
  - Deux couches de convolution 3D (vraie profondeur 3D)
  - Plus de filtres (16 -> 32)
  - Couche cachee plus large
  - Attention conservee
  - Toujours 100 % NumPy (smartphones / PC modestes)

Tous les parametres sont empaquetes dans un seul vecteur float32.
"""

from __future__ import annotations

import numpy as np

from .attention import spatial_temporal_attention


def _pad3d(x: np.ndarray, pad: int) -> np.ndarray:
    if pad == 0:
        return x
    return np.pad(x, [(pad, pad)] * 3, mode="constant")


def conv3d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    """
    x : (C_in, D, H, W)  ou  (D, H, W) pour la premiere couche (C_in=1)
    w : (F, C_in, k, k, k)
    b : (F,)
    Retourne out (F, D, H, W) et cache.
    """
    if x.ndim == 3:
        x = x[np.newaxis, ...]          # (1, D, H, W)
    f_out, c_in, k, _, _ = w.shape
    pad = k // 2
    # pad spatial uniquement
    xp = np.pad(x, [(0, 0)] + [(pad, pad)] * 3, mode="constant")

    # sliding window : (C, D, H, W, k, k, k)
    windows = np.lib.stride_tricks.sliding_window_view(xp, (k, k, k), axis=(1, 2, 3))
    # tensordot sur (C, k, k, k)
    out = np.tensordot(windows, w, axes=([0, 4, 5, 6], [1, 2, 3, 4]))  # (D,H,W,F)
    out = out + b.reshape(1, 1, 1, f_out)
    out = np.transpose(out, (3, 0, 1, 2))  # (F, D, H, W)

    cache = (windows, w.shape, x.shape)
    return out.astype(np.float32), cache


def conv3d_backward(dout: np.ndarray, cache):
    """dout : (F, D, H, W). Retourne dW, db (pas de dx pour simplicite / premiere couches)."""
    windows, w_shape, _ = cache
    f_out = w_shape[0]
    dout_t = np.transpose(dout, (1, 2, 3, 0))  # (D,H,W,F)
    # windows: (C, D, H, W, k, k, k)
    dw = np.tensordot(dout_t, windows, axes=([0, 1, 2], [1, 2, 3]))  # (F, C, k, k, k)
    db = dout_t.reshape(-1, f_out).sum(axis=0)
    return dw.astype(np.float32), db.astype(np.float32)


def _regional_pool_forward(attended_flat: np.ndarray, d: int, h: int, w: int,
                            regions=(2, 2, 2)):
    f_dim = attended_flat.shape[0]
    rd, rh, rw = regions
    bd, bh, bw = max(1, d // rd), max(1, h // rh), max(1, w // rw)
    cd, ch, cw = bd * rd, bh * rh, bw * rw

    vol = attended_flat.reshape(f_dim, d, h, w)[:, :cd, :ch, :cw]
    vol = vol.reshape(f_dim, rd, bd, rh, bh, rw, bw)
    pooled = vol.mean(axis=(2, 4, 6))              # (F, rd, rh, rw)
    pooled_flat = pooled.reshape(f_dim, rd * rh * rw)

    cache = (d, h, w, rd, rh, rw, bd, bh, bw, cd, ch, cw)
    return pooled_flat, cache


def _regional_pool_backward(dpooled_flat: np.ndarray, cache) -> np.ndarray:
    d, h, w, rd, rh, rw, bd, bh, bw, cd, ch, cw = cache
    f_dim = dpooled_flat.shape[0]
    dpooled = dpooled_flat.reshape(f_dim, rd, rh, rw)

    dblock = dpooled[:, :, None, :, None, :, None] / (bd * bh * bw)
    dblock = np.broadcast_to(dblock, (f_dim, rd, bd, rh, bh, rw, bw)).reshape(f_dim, cd, ch, cw)

    darr = np.zeros((f_dim, d, h, w), dtype=np.float32)
    darr[:, :cd, :ch, :cw] = dblock
    return darr


class AttentionCNN3D:
    """
    CNN 3D renforce avec attention.

    Architecture :
        Conv3D-1 (F) -> ReLU -> Conv3D-2 (2F) -> ReLU
        -> Attention -> Pooling regional 2x2x2
        -> FC1 -> ReLU -> FC2 -> Softmax

    Par defaut : F=16 → environ 15-25k parametres (bon compromis performance / volontaires).
    """

    def __init__(self, n_filters: int = 16, kernel_size: int = 3,
                 n_classes: int = 10, use_attention: bool = True,
                 attention_alpha: float = 0.6, n_hidden: int = 64,
                 regions=(2, 2, 2)):
        self.f = n_filters
        self.f2 = n_filters * 2
        self.k = kernel_size
        self.c = n_classes
        self.use_attention = use_attention
        self.attention_alpha = attention_alpha
        self.n_hidden = n_hidden
        self.regions = tuple(regions)
        self.n_regions = int(np.prod(self.regions))
        self.fc_in = self.f2 * self.n_regions

        self.shapes = {
            "conv1_w": (self.f, 1, self.k, self.k, self.k),   # C_in = 1
            "conv1_b": (self.f,),
            "conv2_w": (self.f2, self.f, self.k, self.k, self.k),
            "conv2_b": (self.f2,),
            "fc1_w": (self.fc_in, self.n_hidden),
            "fc1_b": (self.n_hidden,),
            "fc2_w": (self.n_hidden, self.c),
            "fc2_b": (self.c,),
        }
        self._order = ["conv1_w", "conv1_b", "conv2_w", "conv2_b",
                       "fc1_w", "fc1_b", "fc2_w", "fc2_b"]

    def n_params(self) -> int:
        return sum(int(np.prod(s)) for s in self.shapes.values())

    def pack(self, parts: dict) -> np.ndarray:
        return np.concatenate(
            [parts[name].reshape(-1) for name in self._order]
        ).astype(np.float32)

    def unpack(self, flat: np.ndarray) -> dict:
        flat = np.asarray(flat, dtype=np.float32)
        parts, i = {}, 0
        for name in self._order:
            shape = self.shapes[name]
            n = int(np.prod(shape))
            parts[name] = flat[i:i + n].reshape(shape)
            i += n
        return parts

    def init_params(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # He init pour les couches suivies de ReLU
        conv1_w = (rng.normal(0.0, 1.0, self.shapes["conv1_w"])
                   * np.sqrt(2.0 / (1 * self.k ** 3))).astype(np.float32)
        conv1_b = np.zeros(self.shapes["conv1_b"], dtype=np.float32)

        conv2_w = (rng.normal(0.0, 1.0, self.shapes["conv2_w"])
                   * np.sqrt(2.0 / (self.f * self.k ** 3))).astype(np.float32)
        conv2_b = np.zeros(self.shapes["conv2_b"], dtype=np.float32)

        fc1_w = (rng.normal(0.0, 1.0, self.shapes["fc1_w"])
                 * np.sqrt(2.0 / self.fc_in)).astype(np.float32)
        fc1_b = np.zeros(self.shapes["fc1_b"], dtype=np.float32)

        # Xavier pour la derniere couche
        fc2_w = (rng.normal(0.0, 1.0, self.shapes["fc2_w"])
                 * np.sqrt(1.0 / self.n_hidden)).astype(np.float32)
        fc2_b = np.zeros(self.shapes["fc2_b"], dtype=np.float32)

        return self.pack({
            "conv1_w": conv1_w, "conv1_b": conv1_b,
            "conv2_w": conv2_w, "conv2_b": conv2_b,
            "fc1_w": fc1_w, "fc1_b": fc1_b,
            "fc2_w": fc2_w, "fc2_b": fc2_b,
        })

    def forward_backward(self, params_flat: np.ndarray, x: np.ndarray, y: int):
        p = self.unpack(params_flat)

        # --- Conv1 ---
        conv1, cache1 = conv3d_forward(x, p["conv1_w"], p["conv1_b"])
        relu1_mask = conv1 > 0
        relu1 = conv1 * relu1_mask

        # --- Conv2 ---
        conv2, cache2 = conv3d_forward(relu1, p["conv2_w"], p["conv2_b"])
        relu2_mask = conv2 > 0
        relu2 = conv2 * relu2_mask                    # (F2, D, H, W)

        f_dim, d, h, w = relu2.shape
        n_vox = d * h * w
        arr = relu2.reshape(f_dim, n_vox)

        # --- Attention (stop-gradient sur les poids d'attention) ---
        if self.use_attention:
            alpha = self.attention_alpha
            spatial = arr.mean(axis=0, keepdims=True)
            temporal = arr.mean(axis=1, keepdims=True)
            combined = alpha * spatial + (1.0 - alpha) * temporal
            attn = np.tanh(combined).astype(np.float32)
            attended_flat = arr * attn
        else:
            attn = np.ones((f_dim, n_vox), dtype=np.float32)
            attended_flat = arr

        # --- Pooling regional ---
        pooled_flat, cache_pool = _regional_pool_forward(
            attended_flat, d, h, w, regions=self.regions)
        pooled = pooled_flat.reshape(-1)

        # --- Tete dense ---
        z1 = pooled @ p["fc1_w"] + p["fc1_b"]
        h1_mask = z1 > 0
        hidden = z1 * h1_mask

        logits = hidden @ p["fc2_w"] + p["fc2_b"]
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / (exp.sum() + 1e-12)
        loss = float(-np.log(probs[y] + 1e-9))
        correct = int(np.argmax(probs) == y)

        # ========== BACKWARD ==========
        dlogits = probs.copy()
        dlogits[y] -= 1.0

        dfc2_w = np.outer(hidden, dlogits).astype(np.float32)
        dfc2_b = dlogits.astype(np.float32)
        dhidden = p["fc2_w"] @ dlogits

        dz1 = (dhidden * h1_mask).astype(np.float32)
        dfc1_w = np.outer(pooled, dz1).astype(np.float32)
        dfc1_b = dz1
        dpooled = p["fc1_w"] @ dz1

        dpooled_flat = dpooled.reshape(f_dim, self.n_regions)
        dattended = _regional_pool_backward(dpooled_flat, cache_pool)
        # attention en stop-gradient : le gradient passe comme si attn=constante
        darr = (dattended.reshape(f_dim, n_vox) * attn).reshape(f_dim, d, h, w)

        drelu2 = darr * relu2_mask
        dconv2_w, dconv2_b = conv3d_backward(drelu2, cache2)

        # Pour garder le calcul leger sur volontaires, on ne propage pas dx jusqu'a conv1
        # (approximation courante sur premiere couche). On calcule quand meme dconv1 via relu1.
        # Approximation : on utilise la moyenne des gradients spatiaux comme signal.
        drelu1 = np.mean(drelu2, axis=0, keepdims=True)  # signal simplifie
        drelu1 = np.broadcast_to(drelu1, relu1.shape) * relu1_mask
        dconv1_w, dconv1_b = conv3d_backward(drelu1, cache1)

        grad = self.pack({
            "conv1_w": dconv1_w, "conv1_b": dconv1_b,
            "conv2_w": dconv2_w, "conv2_b": dconv2_b,
            "fc1_w": dfc1_w, "fc1_b": dfc1_b,
            "fc2_w": dfc2_w, "fc2_b": dfc2_b,
        })
        return grad, loss, correct, {"probs": probs}

    def predict(self, params_flat: np.ndarray, x: np.ndarray) -> int:
        p = self.unpack(params_flat)
        conv1, _ = conv3d_forward(x, p["conv1_w"], p["conv1_b"])
        relu1 = np.maximum(conv1, 0)
        conv2, _ = conv3d_forward(relu1, p["conv2_w"], p["conv2_b"])
        relu2 = np.maximum(conv2, 0)
        f_dim, d, h, w = relu2.shape
        arr = relu2.reshape(f_dim, -1)
        if self.use_attention:
            alpha = self.attention_alpha
            spatial = arr.mean(axis=0, keepdims=True)
            temporal = arr.mean(axis=1, keepdims=True)
            attn = np.tanh(alpha * spatial + (1 - alpha) * temporal)
            arr = arr * attn
        pooled_flat, _ = _regional_pool_forward(arr, d, h, w, regions=self.regions)
        pooled = pooled_flat.reshape(-1)
        hidden = np.maximum(pooled @ p["fc1_w"] + p["fc1_b"], 0)
        logits = hidden @ p["fc2_w"] + p["fc2_b"]
        return int(np.argmax(logits))
