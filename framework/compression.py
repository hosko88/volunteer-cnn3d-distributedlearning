"""
Transport et compression des gradients (version optimisée)
==========================================================
Pour alléger les échanges volontaire <-> serveur (réseaux mobiles, Wi-Fi faible),
on compresse les gradients de plusieurs façons combinables :

  1. Quantification :
       - fp32  : aucune
       - fp16  : x2
       - int8  : x4  (avec échelle dynamique par vecteur)
  2. Sparsification top-k :
       on ne transmet que les k% de coefficients de plus grande amplitude.
       Les indices sont stockés en uint16 si possible (n < 65536).
  3. Compression zlib (niveau 9) + base64 pour le transport HTTP.

Optionnel (activable) : Error Feedback (EF)
  On conserve l'erreur de quantification/sparsification côté volontaire
  et on l'ajoute au prochain gradient. Cela stabilise fortement la convergence
  quand topk est bas (ex. 0.1 – 0.3).

Le format reste rétro-compatible avec les anciennes versions (mode dense/sparse).
"""

from __future__ import annotations

import json
import zlib
import base64
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _dtype_np(name: str):
    if name == "fp16":
        return np.float16
    if name == "int8":
        return np.int8
    return np.float32


def _pack_indices(idx: np.ndarray, n: int) -> Tuple[bytes, str]:
    """Choisit uint16 ou uint32 selon la taille du modèle."""
    if n <= 65535:
        return idx.astype("<u2").tobytes(), "u16"
    return idx.astype("<u4").tobytes(), "u32"


def _unpack_indices(blob: bytes, k: int, idx_dtype: str) -> np.ndarray:
    if idx_dtype == "u16":
        return np.frombuffer(blob, dtype="<u2", count=k)
    return np.frombuffer(blob, dtype="<u4", count=k)


# --------------------------------------------------------------------------- #
#  Quantification int8 avec échelle
# --------------------------------------------------------------------------- #

def _quantize_int8(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Quantification symétrique int8.
    Retourne (q, scale) tel que approx = q.astype(float32) * scale
    """
    absmax = float(np.max(np.abs(vec))) + 1e-12
    scale = absmax / 127.0
    q = np.clip(np.round(vec / scale), -127, 127).astype(np.int8)
    return q, scale


def _dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


# --------------------------------------------------------------------------- #
#  API principale
# --------------------------------------------------------------------------- #

def encode_vector(
    vec,
    dtype: str = "fp16",
    topk: float = 1.0,
    residual: Optional[np.ndarray] = None,
) -> Tuple[str, int, Optional[np.ndarray]]:
    """
    Encode un vecteur (gradient) -> (payload base64, taille_octets_compressés, nouveau_residual)

    Parameters
    ----------
    vec : array-like
        Gradient à encoder.
    dtype : "fp32" | "fp16" | "int8"
    topk : float in (0, 1]
        Fraction des coefficients de plus grande amplitude à transmettre.
    residual : np.ndarray or None
        Erreur de compression précédente (Error Feedback). Si fourni, on l'ajoute
        à vec avant compression et on retourne le nouveau residual.

    Returns
    -------
    payload : str
        Chaîne base64 prête à être envoyée en HTTP.
    nbytes : int
        Taille en octets du blob compressé (utile pour les métriques).
    new_residual : np.ndarray or None
        Nouveau residual à conserver côté volontaire (None si residual=None).
    """
    vec = np.asarray(vec, dtype=np.float32).ravel()
    n = int(vec.size)

    # --- Error Feedback ---
    if residual is not None:
        residual = np.asarray(residual, dtype=np.float32).ravel()
        if residual.shape != vec.shape:
            residual = np.zeros_like(vec)
        work = vec + residual
    else:
        work = vec

    header = {
        "mode": "dense" if topk >= 1.0 else "sparse",
        "dtype": dtype,
        "n": n,
    }

    if topk >= 1.0:
        # Dense
        if dtype == "int8":
            q, scale = _quantize_int8(work)
            header["scale"] = scale
            blob = q.tobytes()
            approx = _dequantize_int8(q, scale)
        else:
            npdt = _dtype_np(dtype)
            blob = work.astype(npdt).tobytes()
            approx = work.astype(npdt).astype(np.float32)
    else:
        # Sparse top-k
        k = max(1, int(n * float(topk)))
        # argpartition est O(n) et plus rapide que argsort
        idx = np.argpartition(np.abs(work), n - k)[n - k :]
        idx.sort()  # pour un packing plus compressable

        vals = work[idx]

        if dtype == "int8":
            q, scale = _quantize_int8(vals)
            header["scale"] = scale
            header["k"] = int(k)
            idx_bytes, idx_dtype = _pack_indices(idx, n)
            header["idx_dtype"] = idx_dtype
            blob = idx_bytes + q.tobytes()
            approx = np.zeros(n, dtype=np.float32)
            approx[idx] = _dequantize_int8(q, scale)
        else:
            npdt = _dtype_np(dtype)
            header["k"] = int(k)
            idx_bytes, idx_dtype = _pack_indices(idx, n)
            header["idx_dtype"] = idx_dtype
            blob = idx_bytes + vals.astype(npdt).tobytes()
            approx = np.zeros(n, dtype=np.float32)
            approx[idx] = vals.astype(npdt).astype(np.float32)

    # Residual pour la prochaine itération
    new_residual = None
    if residual is not None:
        new_residual = (work - approx).astype(np.float32)

    # Empaquetage final : [4 octets longueur header][header JSON][blob]
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw = len(hjson).to_bytes(4, "little") + hjson + blob

    # zlib niveau 9 (meilleure compression, le modèle est petit donc le coût CPU est négligeable)
    comp = zlib.compress(raw, level=9)
    payload = base64.b64encode(comp).decode("ascii")
    return payload, len(comp), new_residual


def decode_vector(payload: str) -> np.ndarray:
    """
    Décode une chaîne base64 -> vecteur dense float32.
    Compatible avec l'ancien format et le nouveau (int8 + idx_dtype).
    """
    comp = base64.b64decode(payload.encode("ascii"))
    raw = zlib.decompress(comp)

    hlen = int.from_bytes(raw[:4], "little")
    header = json.loads(raw[4 : 4 + hlen].decode("utf-8"))
    blob = raw[4 + hlen :]

    n = int(header["n"])
    dtype = header["dtype"]
    mode = header["mode"]

    if mode == "dense":
        if dtype == "int8":
            q = np.frombuffer(blob, dtype=np.int8, count=n)
            scale = float(header.get("scale", 1.0))
            return _dequantize_int8(q, scale)
        npdt = _dtype_np(dtype)
        return np.frombuffer(blob, dtype=npdt).astype(np.float32).copy()

    # Sparse
    k = int(header["k"])
    idx_dtype = header.get("idx_dtype", "u4")  # rétro-compatibilité
    idx_size = 2 if idx_dtype == "u16" else 4
    idx_bytes = idx_size * k

    idx = _unpack_indices(blob[:idx_bytes], k, idx_dtype)

    if dtype == "int8":
        q = np.frombuffer(blob[idx_bytes:], dtype=np.int8, count=k)
        scale = float(header.get("scale", 1.0))
        vals = _dequantize_int8(q, scale)
    else:
        npdt = _dtype_np(dtype)
        vals = np.frombuffer(blob[idx_bytes:], dtype=npdt).astype(np.float32)

    out = np.zeros(n, dtype=np.float32)
    out[idx] = vals
    return out


def raw_size_bytes(n_params: int, dtype: str = "fp32") -> int:
    """Taille brute (sans sparsification ni zlib) d'un vecteur."""
    if dtype == "int8":
        return n_params * 1
    if dtype == "fp16":
        return n_params * 2
    return n_params * 4


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Ratio de compression (original / compressé)."""
    if compressed_bytes <= 0:
        return 0.0
    return original_bytes / compressed_bytes


# --------------------------------------------------------------------------- #
#  Test rapide (exécutable directement)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    v = rng.standard_normal(1434).astype(np.float32) * 0.01

    print("=== Benchmark compression (n=1434) ===\n")
    raw = raw_size_bytes(len(v), "fp32")
    print(f"Taille brute fp32 : {raw} octets\n")

    for dtype in ("fp16", "int8"):
        for topk in (1.0, 0.5, 0.25, 0.1):
            payload, nbytes, _ = encode_vector(v, dtype=dtype, topk=topk)
            recovered = decode_vector(payload)
            err = np.linalg.norm(v - recovered) / (np.linalg.norm(v) + 1e-12)
            ratio = compression_ratio(raw, nbytes)
            print(
                f"  {dtype:4s}  topk={topk:.2f}  "
                f"→ {nbytes:5d} o  (x{ratio:5.1f})  "
                f"erreur relative = {err:.4f}"
            )
        print()
