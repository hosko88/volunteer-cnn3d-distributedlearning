import numpy as np


def spatial_temporal_attention(features, alpha=0.5):
    """Attention spatiale et temporelle simple, compatible NumPy pur."""
    arr = np.asarray(features, dtype=np.float32)

    if arr.ndim == 1:
        return np.tanh(arr).astype(np.float32)

    if arr.ndim == 2:
        spatial = np.mean(arr, axis=0, keepdims=True)
        temporal = np.mean(arr, axis=1, keepdims=True)
        combined = alpha * spatial + (1 - alpha) * temporal
        weights = np.tanh(combined)
        return weights.astype(np.float32)

    spatial = np.mean(arr, axis=tuple(range(arr.ndim - 1)), keepdims=True)
    temporal = np.mean(arr, axis=tuple(range(1, arr.ndim)), keepdims=True)
    combined = alpha * spatial + (1 - alpha) * temporal
    weights = np.tanh(combined)
    return weights.astype(np.float32)
