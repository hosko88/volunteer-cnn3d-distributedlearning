"""
Fournisseurs de donnees pour les jobs CNN 3D
=============================================
Chaque "DataProvider" sait fournir :
  - un lot d'entrainement : sample_batch(rng, batch_size) -> (X, y)
      X : (batch_size, D, H, W) float32, volumes normalises dans [0,1]
      y : (batch_size,) int, labels de classe
  - un lot d'evaluation   : sample_eval(n) -> (X, y)  (deterministe, sous-ensemble fixe)

Deux implementations :
  - CIFAR10Provider   : donnees REELLES (Phase 1). Telechargement automatique
                        (pas de dependance a torchvision : lecture directe du
                        format officiel cifar-10-python, cf. prepare_data.py).
  - SyntheticProvider : donnees aleatoires deterministes, utilisees UNIQUEMENT
                        par les tests unitaires (pas de reseau, execution
                        instantanee). Ne jamais utiliser en execution reelle :
                        c'est exactement ce que ce projet cherchait a corriger.

Extension prevue pour les phases suivantes (voir memoire, section phases) :
  - ModelNet40Provider (Phase 2, objets 3D)
  - ShapeNetProvider   (Phase 3, formes/scenes 3D)
  Ces deux classes sont esquissees plus bas avec des instructions claires ;
  elles necessitent des donnees volumetriques natives (voxels) que l'etudiant
  doit deposer localement (cf. README pour la procedure de telechargement,
  qui requiert un compte pour ShapeNet).
"""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MIRROR_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"


class DataProvider:
    """Contrat minimal attendu par les jobs CNN 3D."""

    n_classes: int
    volume_shape: tuple

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        raise NotImplementedError

    def sample_eval(self, n: int):
        raise NotImplementedError

    def get_shard(self, shard_id: int, n_shards: int):
        """
        Retourne (x_shard, y_shard) : la partition shard_id parmi n_shards des
        donnees d'ENTRAINEMENT UNIQUEMENT (jamais les donnees de test/eval,
        qui restent cote serveur pour une evaluation homogene).

        Permet a un volontaire de recevoir sa partition directement du
        serveur (via le protocole existant) SANS telecharger le jeu de
        donnees complet lui-meme -- seul le serveur (ou une machine qui
        prepare les donnees une fois) a besoin du jeu complet en local.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Phase 1 : CIFAR-10 (donnees reelles)
# --------------------------------------------------------------------------- #

def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _download_with_progress(url: str, dest: Path, timeout: int = 20) -> None:
    """Telechargement avec barre de progression et timeout de connexion,
    pour eviter l'impression de blocage silencieux."""
    import socket
    import time

    socket.setdefaulttimeout(timeout)
    start = time.time()
    last_print = [0.0]

    def _hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        now = time.time()
        if now - last_print[0] < 0.5 and downloaded < total_size:
            return  # limite l'affichage a 2x/seconde
        last_print[0] = now
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            elapsed = now - start
            speed = mb_done / elapsed if elapsed > 0 else 0
            print(f"\r[cifar10] {pct:5.1f}%  ({mb_done:6.1f} / {mb_total:6.1f} Mo)  "
                  f"{speed:5.2f} Mo/s", end="", flush=True)
        else:
            print(f"\r[cifar10] {downloaded / (1024*1024):.1f} Mo telecharges...",
                  end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()  # nouvelle ligne apres la barre de progression


def download_cifar10(root: Path = DATA_ROOT / "cifar10") -> Path:
    """Telecharge et extrait CIFAR-10 (format python officiel) si absent."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-10-batches-py"
    if extracted.is_dir() and any(extracted.iterdir()):
        return extracted

    archive = root / "cifar-10-python.tar.gz"
    if not archive.exists():
        tmp = archive.with_suffix(".part")
        try:
            print(f"[cifar10] telechargement depuis {CIFAR10_URL}")
            print("[cifar10] (~170 Mo -- peut prendre plusieurs minutes selon la connexion)")
            _download_with_progress(CIFAR10_URL, tmp)
        except Exception as e:
            print(f"\n[cifar10] echec sur la source principale ({e}).")
            print(f"[cifar10] nouvelle tentative via le mirroir : {CIFAR10_MIRROR_URL}")
            _download_with_progress(CIFAR10_MIRROR_URL, tmp)
        tmp.rename(archive)
        print("[cifar10] telechargement termine.")

    print("[cifar10] extraction ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    print("[cifar10] extraction terminee.")
    return extracted


def load_cifar10_raw(root: Path = DATA_ROOT / "cifar10", split: str = "train"):
    """Retourne (X, y) : X (N,3,32,32) uint8, y (N,) int64."""
    extracted = download_cifar10(root)
    batches = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]


    xs, ys = [], []
    for b in batches:
        d = _unpickle(extracted / b)
        xs.append(d[b"data"])
        ys.append(d[b"labels"])
    x = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.uint8)
    y = np.concatenate([np.asarray(v) for v in ys], axis=0).astype(np.int64)
    return x, y


def image_to_volume(img_u8: np.ndarray, volume_shape=(8, 8, 8)) -> np.ndarray:
    """
    Convertit une image CIFAR-10 (3,32,32) uint8 en volume pseudo-3D (D,H,W)
    float32 dans [0,1] : fusion des canaux RGB (niveaux de gris), reduction
    spatiale par moyennage de blocs vers (H,W), puis extrusion sur la
    profondeur D. C'est une passerelle deliberement simple entre une image 2D
    et l'entree volumetrique du CNN 3D pour la Phase 1 (donnees plus simples,
    cf. memoire) ; les Phases 2 et 3 utiliseront des voxels natifs.
    """
    d, h, w = volume_shape
    img = img_u8.astype(np.float32) / 255.0
    gray = img.mean(axis=0)  # (32,32)

    bh, bw = max(1, 32 // h), max(1, 32 // w)
    cropped = gray[: bh * h, : bw * w]
    pooled = cropped.reshape(h, bh, w, bw).mean(axis=(1, 3))  # (h,w)

    volume = np.stack([pooled * (1.0 - 0.02 * i) for i in range(d)], axis=0)
    return volume.astype(np.float32)


class CIFAR10Provider(DataProvider):
    """Phase 1 : CIFAR-10 reel, converti en volumes pseudo-3D."""

    def __init__(self, root: Path = DATA_ROOT / "cifar10",
                 volume_shape=(8, 8, 8), n_classes: int = 10):
        self.root = Path(root)
        self.volume_shape = tuple(volume_shape)
        self.n_classes = n_classes
        self._train = None
        self._test = None

    def _ensure_train(self):
        if self._train is None:
            self._train = load_cifar10_raw(self.root, split="train")
        return self._train

    def _ensure_test(self):
        if self._test is None:
            self._test = load_cifar10_raw(self.root, split="test")
        return self._test

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        x_raw, y_raw = self._ensure_train()
        idx = rng.integers(0, len(x_raw), size=batch_size)
        volumes = np.stack([image_to_volume(x_raw[i], self.volume_shape) for i in idx])
        labels = y_raw[idx] % self.n_classes
        return volumes, labels

    def sample_eval(self, n: int):
        x_raw, y_raw = self._ensure_test()
        n = min(n, len(x_raw))
        idx = np.arange(n)  # sous-ensemble fixe et deterministe
        volumes = np.stack([image_to_volume(x_raw[i], self.volume_shape) for i in idx])
        labels = y_raw[idx] % self.n_classes
        return volumes, labels

    def get_shard(self, shard_id: int, n_shards: int):
        x_raw, y_raw = self._ensure_train()
        # decoupage entrelace (pas par blocs contigus) : chaque partition
        # couvre une diversite de classes similaire, evite qu'une partition
        # ne tombe sur une plage non representative du jeu de donnees.
        idx = np.arange(shard_id, len(x_raw), n_shards)
        volumes = np.stack([image_to_volume(x_raw[i], self.volume_shape) for i in idx])
        labels = y_raw[idx] % self.n_classes
        return volumes.astype(np.float32), labels.astype(np.int64)


# --------------------------------------------------------------------------- #
#  Phases 2 et 3 : donnees volumetriques natives (a completer par l'etudiant)
# --------------------------------------------------------------------------- #

class VoxelFolderProvider(DataProvider):
    """
    Fournisseur generique pour des donnees deja converties en voxels et
    enregistrees localement sous forme de tableaux .npy :
        data/<nom>/train_x.npy  (N,D,H,W) float32 dans [0,1]
        data/<nom>/train_y.npy  (N,) int
        data/<nom>/test_x.npy, data/<nom>/test_y.npy
    Utilisable tel quel pour ModelNet40 (Phase 2) et ShapeNet (Phase 3) une
    fois les modeles convertis en voxels (voir README pour la procedure de
    telechargement et de voxelisation, hors-scope de ce fournisseur).
    """

    def __init__(self, root: Path, volume_shape=(8, 8, 8), n_classes: int = 10):
        self.root = Path(root)
        self.volume_shape = tuple(volume_shape)
        self.n_classes = n_classes
        self._train = None
        self._test = None

    def _load(self, split):
        xp = self.root / f"{split}_x.npy"
        yp = self.root / f"{split}_y.npy"
        if not xp.exists() or not yp.exists():
            raise FileNotFoundError(
                f"Donnees introuvables pour '{self.root.name}' ({split}). "
                f"Attendu : {xp} et {yp}. Voir README pour la procedure de "
                f"preparation (telechargement + voxelisation)."
            )
        return np.load(xp).astype(np.float32), np.load(yp).astype(np.int64)

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        if self._train is None:
            self._train = self._load("train")
        x, y = self._train
        idx = rng.integers(0, len(x), size=batch_size)
        return x[idx], y[idx] % self.n_classes

    def sample_eval(self, n: int):
        if self._test is None:
            self._test = self._load("test")
        x, y = self._test
        n = min(n, len(x))
        return x[:n], y[:n] % self.n_classes

    def get_shard(self, shard_id: int, n_shards: int):
        if self._train is None:
            self._train = self._load("train")
        x, y = self._train
        idx = np.arange(shard_id, len(x), n_shards)
        return x[idx].astype(np.float32), (y[idx] % self.n_classes).astype(np.int64)


def modelnet40_provider(volume_shape=(8, 8, 8), n_classes: int = 40) -> VoxelFolderProvider:
    return VoxelFolderProvider(DATA_ROOT / "modelnet40_voxels", volume_shape, n_classes)


def shapenet_provider(volume_shape=(8, 8, 8), n_classes: int = 16) -> VoxelFolderProvider:
    return VoxelFolderProvider(DATA_ROOT / "shapenet_voxels", volume_shape, n_classes)


# --------------------------------------------------------------------------- #
#  Fournisseur cote VOLONTAIRE : partition recue du serveur, sans telechargement
# --------------------------------------------------------------------------- #

class ShardProvider(DataProvider):
    """
    Fournisseur cote VOLONTAIRE : encapsule uniquement la partition de
    donnees recue du serveur (jamais le jeu de donnees complet). Aucun
    telechargement -- les donnees sont deja en memoire, transmises une fois
    via le protocole reseau existant (route /shard).
    """

    def __init__(self, x_shard: np.ndarray, y_shard: np.ndarray,
                 volume_shape=None, n_classes: int = 10):
        self.x_shard = np.asarray(x_shard, dtype=np.float32)
        self.y_shard = np.asarray(y_shard, dtype=np.int64)
        self.volume_shape = volume_shape or self.x_shard.shape[1:]
        self.n_classes = n_classes

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        idx = rng.integers(0, len(self.x_shard), size=batch_size)
        return self.x_shard[idx], self.y_shard[idx]

    def sample_eval(self, n: int):
        # un volontaire n'evalue pas (l'evaluation reste server-side sur les
        # vraies donnees de test) ; fourni pour respecter le contrat.
        n = min(n, len(self.x_shard))
        return self.x_shard[:n], self.y_shard[:n]


# --------------------------------------------------------------------------- #
#  Fournisseur synthetique : TESTS UNITAIRES UNIQUEMENT
# --------------------------------------------------------------------------- #

class SyntheticProvider(DataProvider):
    """
    Donnees aleatoires deterministes (graine fixe), rapides, sans reseau.
    Reserve aux tests automatises : ne jamais utiliser pour une execution
    reelle du serveur ou d'un volontaire (cf. AttentionCNN3DJob /
    CNN3DJob, ou le fournisseur reel est celui utilise par defaut).
    """

    def __init__(self, volume_shape=(6, 6, 6), n_classes: int = 2, seed: int = 0):
        self.volume_shape = tuple(volume_shape)
        self.n_classes = n_classes
        self._rng = np.random.default_rng(seed)

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        x = rng.normal(0.0, 1.0, (batch_size, *self.volume_shape)).astype(np.float32)
        y = rng.integers(0, self.n_classes, size=batch_size)
        return x, y

    def sample_eval(self, n: int):
        x = self._rng.normal(0.0, 1.0, (n, *self.volume_shape)).astype(np.float32)
        y = self._rng.integers(0, self.n_classes, size=n)
        return x, y


class LearnableSyntheticProvider(DataProvider):
    """
    Comme SyntheticProvider, mais avec un VRAI signal appris possible
    (label derive deterministiquement de la moyenne du volume). Reserve
    aux tests de convergence (verifier qu'un optimiseur progresse
    reellement), la ou SyntheticProvider (bruit pur, x et y independants)
    ne permet pas d'evaluer la capacite d'apprentissage -- un test qui
    exige une baisse de perte sur un seul pas de bruit pur est fragile
    par construction, quel que soit l'optimiseur utilise.
    """

    def __init__(self, volume_shape=(6, 6, 6), n_classes: int = 2, seed: int = 0):
        self.volume_shape = tuple(volume_shape)
        self.n_classes = n_classes
        self._rng_eval = np.random.default_rng(seed + 999)

    def _make(self, rng: np.random.Generator, n: int):
        x = rng.normal(0.0, 1.0, (n, *self.volume_shape)).astype(np.float32)
        means = x.reshape(n, -1).mean(axis=1)
        span = means.max() - means.min() + 1e-8
        buckets = (means - means.min()) / span
        y = (buckets * self.n_classes).astype(np.int64).clip(0, self.n_classes - 1)
        return x, y

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        return self._make(rng, batch_size)

    def sample_eval(self, n: int):
        return self._make(self._rng_eval, n)
