"""
Vrais datasets pour la progression 2D → 3D
==========================================
Phase 1 : CIFAR-10 officiel (torchvision)
Phase 2/3 : ModelNet40 officiel (point clouds Stanford HDF5 → voxels 32³)

Sources :
  - CIFAR-10 : https://www.cs.toronto.edu/~kriz/cifar.html
  - ModelNet40 : https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
                (jeu standard utilisé par PointNet, DGCNN, etc.)
"""

from __future__ import annotations

import os
import glob
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

MODELNET_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
MODELNET_DIR = DATA_ROOT / "modelnet40_ply_hdf5_2048"
VOXEL_CACHE = DATA_ROOT / "modelnet40_voxels32"


# --------------------------------------------------------------------------- #
#  Phase 1 — CIFAR-10 (vrai dataset)
# --------------------------------------------------------------------------- #
def get_cifar10(train: bool = True):
    from torchvision import datasets, transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return datasets.CIFAR10(
        root=str(DATA_ROOT / "cifar10"),
        train=train,
        download=True,
        transform=tfm,
    )


class CIFAR10Provider:
    """Vrai CIFAR-10 (50 000 train / 10 000 test)."""

    def __init__(self, n_classes: int = 10):
        self.n_classes = n_classes
        self._train = None
        self._test = None

    def _ensure(self):
        if self._train is None:
            print("[CIFAR-10] Chargement / telechargement du dataset officiel...")
            self._train = get_cifar10(train=True)
            self._test = get_cifar10(train=False)
            print(f"[CIFAR-10] train={len(self._train)} test={len(self._test)}")

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        self._ensure()
        n = len(self._train)
        indices = rng.integers(0, n, size=batch_size)
        xs, ys = [], []
        for i in indices:
            x, y = self._train[int(i)]
            xs.append(x.numpy())
            ys.append(y)
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)

    def sample_eval(self, n: int = 1000):
        self._ensure()
        n = min(n, len(self._test))
        xs, ys = [], []
        for i in range(n):
            x, y = self._test[i]
            xs.append(x.numpy())
            ys.append(y)
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)

    def get_shard(self, shard_id: int, n_shards: int):
        """Partition d'ENTRAINEMENT (jamais le test) -- permet a un
        volontaire de recevoir uniquement sa part depuis le serveur, sans
        telecharger le dataset complet (~170 Mo) lui-meme."""
        self._ensure()
        n = len(self._train)
        idx = np.arange(shard_id, n, n_shards)  # decoupage entrelace
        xs, ys = [], []
        for i in idx:
            x, y = self._train[int(i)]
            xs.append(x.numpy())
            ys.append(y)
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  Phase 2/3 — ModelNet40 REEL (point clouds → voxels)
# --------------------------------------------------------------------------- #
def _download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    print(f"[download] {url}")
    print(f"         → {dest}")

    def _progress(block, block_size, total):
        if total > 0 and block % 50 == 0:
            pct = min(100.0, block * block_size * 100.0 / total)
            print(f"\r  {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()


def _ensure_modelnet40_hdf5():
    """Telecharge le ModelNet40 officiel (point clouds HDF5, Stanford)."""
    if MODELNET_DIR.exists() and list(MODELNET_DIR.glob("*.h5")):
        return MODELNET_DIR

    zip_path = DATA_ROOT / "modelnet40_ply_hdf5_2048.zip"
    try:
        _download_file(MODELNET_URL, zip_path)
    except Exception as e:
        # Miroir alternatif parfois utilise
        alt = "https://huggingface.co/datasets/caidas/neuma-modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip"
        print(f"[ModelNet40] Echec URL principale ({e}), essai miroir...")
        try:
            _download_file(alt, zip_path)
        except Exception as e2:
            raise RuntimeError(
                "Impossible de telecharger ModelNet40. "
                "Telechargez manuellement :\n"
                f"  {MODELNET_URL}\n"
                f"et extrayez dans {DATA_ROOT}/"
            ) from e2

    print("[ModelNet40] Extraction...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_ROOT)
    # Selon les archives, le dossier peut etre a la racine ou imbrique
    if not MODELNET_DIR.exists():
        candidates = list(DATA_ROOT.glob("**/modelnet40_ply_hdf5_2048"))
        if candidates:
            pass  # deja au bon endroit via extract
    if not list(Path(DATA_ROOT).rglob("ply_data_train*.h5")) and not list(MODELNET_DIR.glob("*.h5")):
        raise RuntimeError("Extraction ModelNet40 incomplete (fichiers .h5 introuvables)")
    return MODELNET_DIR


def _load_modelnet_points(partition: str):
    """Charge les point clouds ModelNet40 (N, 2048, 3) + labels."""
    try:
        import h5py
    except ImportError:
        raise ImportError("Installez h5py : pip install h5py")

    _ensure_modelnet40_hdf5()
    # Cherche les h5 dans le dossier standard ou sous-dossiers
    patterns = [
        str(MODELNET_DIR / f"*{partition}*.h5"),
        str(DATA_ROOT / "**" / f"*{partition}*.h5"),
        str(DATA_ROOT / "modelnet40_ply_hdf5_2048" / f"*{partition}*.h5"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier HDF5 ModelNet40 ({partition}) trouve sous {DATA_ROOT}"
        )

    all_data, all_label = [], []
    for h5_name in files:
        with h5py.File(h5_name, "r") as f:
            all_data.append(f["data"][:].astype(np.float32))
            all_label.append(f["label"][:].astype(np.int64).reshape(-1))
    data = np.concatenate(all_data, axis=0)   # (N, 2048, 3)
    label = np.concatenate(all_label, axis=0)
    print(f"[ModelNet40] {partition}: {len(data)} objets, shape points={data.shape}")
    return data, label


def _points_to_voxel(points: np.ndarray, resolution: int = 32) -> np.ndarray:
    """
    Voxelise un nuage de points (M, 3) en grille (R, R, R) binaire/occupancy.
    Methode : normalise dans [0,1]^3 puis histogramme 3D.
    """
    pts = points.copy()
    # Centrer
    pts = pts - pts.mean(axis=0, keepdims=True)
    # Scale dans [-0.5, 0.5] approximatif
    scale = np.abs(pts).max() + 1e-6
    pts = pts / (2.0 * scale) + 0.5  # [0, 1]
    pts = np.clip(pts, 0.0, 1.0 - 1e-6)

    idx = np.floor(pts * resolution).astype(np.int32)
    voxel = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    voxel[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return voxel


def _build_voxel_cache(resolution: int = 32):
    """Convertit tout ModelNet40 en voxels et met en cache .npy."""
    VOXEL_CACHE.mkdir(parents=True, exist_ok=True)
    marker = VOXEL_CACHE / "ready.flag"
    if marker.exists() and (VOXEL_CACHE / "train_x.npy").exists():
        return VOXEL_CACHE

    print("[ModelNet40] Voxelisation 32³ des vrais point clouds (une seule fois)...")
    train_pts, train_y = _load_modelnet_points("train")
    test_pts, test_y = _load_modelnet_points("test")

    def convert(pts_arr):
        out = np.zeros((len(pts_arr), resolution, resolution, resolution), dtype=np.float32)
        for i in range(len(pts_arr)):
            out[i] = _points_to_voxel(pts_arr[i], resolution)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(pts_arr)}")
        return out

    train_x = convert(train_pts)
    test_x = convert(test_pts)

    np.save(VOXEL_CACHE / "train_x.npy", train_x)
    np.save(VOXEL_CACHE / "train_y.npy", train_y.astype(np.int64))
    np.save(VOXEL_CACHE / "test_x.npy", test_x)
    np.save(VOXEL_CACHE / "test_y.npy", test_y.astype(np.int64))
    marker.write_text(
        f"ModelNet40 officiel voxelise {resolution}^3\n"
        f"train={len(train_x)} test={len(test_x)}\n"
        f"source={MODELNET_URL}\n"
    )
    print(f"[ModelNet40] Cache voxels pret : {VOXEL_CACHE}")
    return VOXEL_CACHE


class ModelNet40Provider:
    """
    Vrai ModelNet40 (40 classes).
    Source : point clouds officiels Stanford → voxels 32×32×32.
    """

    def __init__(self, n_classes: int = 40, resolution: int = 32):
        self.n_classes = n_classes
        self.resolution = resolution
        self._train_x = None
        self._train_y = None
        self._test_x = None
        self._test_y = None

    def _ensure(self):
        if self._train_x is None:
            cache = _build_voxel_cache(self.resolution)
            self._train_x = np.load(cache / "train_x.npy")
            self._train_y = np.load(cache / "train_y.npy")
            self._test_x = np.load(cache / "test_x.npy")
            self._test_y = np.load(cache / "test_y.npy")
            print(f"[ModelNet40] charge train={len(self._train_x)} test={len(self._test_x)}")

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        self._ensure()
        n = len(self._train_x)
        indices = rng.integers(0, n, size=batch_size)
        x = self._train_x[indices][:, np.newaxis, ...]  # (B, 1, D, H, W)
        y = self._train_y[indices]
        return x.astype(np.float32), y.astype(np.int64)

    def sample_eval(self, n: int = 800):
        self._ensure()
        n = min(n, len(self._test_x))
        x = self._test_x[:n][:, np.newaxis, ...]
        y = self._test_y[:n]
        return x.astype(np.float32), y.astype(np.int64)

    def get_shard(self, shard_id: int, n_shards: int):
        """Partition d'ENTRAINEMENT (jamais le test) -- evite a un volontaire
        de devoir telecharger + voxeliser ModelNet40 lui-meme (operation
        lente : nuages de points -> voxels 32^3)."""
        self._ensure()
        idx = np.arange(shard_id, len(self._train_x), n_shards)
        x = self._train_x[idx][:, np.newaxis, ...]
        y = self._train_y[idx]
        return x.astype(np.float32), y.astype(np.int64)


# Alias pour compatibilite
OrganMNIST3DProvider = ModelNet40Provider  # si besoin medical plus tard
