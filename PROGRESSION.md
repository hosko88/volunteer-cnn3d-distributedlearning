# Progression 2D → 3D — Vrais datasets

## Datasets

| Phase | Dataset | Source | Type | Classes |
|-------|---------|--------|------|---------|
| **1** | **CIFAR-10** | Toronto / torchvision (officiel) | Images 2D 32×32 RGB | 10 |
| **2** | **ModelNet40** | Stanford ShapeNet HDF5 (officiel) | Voxels 32³ (depuis point clouds) | 40 |
| **3** | **ModelNet40** + Attention | Idem | Voxels 32³ | 40 |

### Sources exactes
- CIFAR-10 : https://www.cs.toronto.edu/~kriz/cifar.html
- ModelNet40 : https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip  
  (même fichier que PointNet, DGCNN, etc.)

La voxelisation (point cloud → grille 32³) est faite **une seule fois** au premier lancement de la Phase 2 ou 3, puis mise en cache dans `data/modelnet40_voxels32/`.

## Installation

```bash
pip install -r requirements-server.txt
# inclut : torch, torchvision, h5py
```

## Lancement

```bash
# Phase 1 — vrai CIFAR-10
python scripts/run_server.py --host 0.0.0.0 --phase 1

# Phase 2 — vrai ModelNet40 (3D)
python scripts/run_server.py --host 0.0.0.0 --phase 2

# Phase 3 — ModelNet40 + Attention
python scripts/run_server.py --host 0.0.0.0 --phase 3
```

Volontaire :
```bash
python scripts/run_volunteer.py --server http://IP:5000 --device mon-pc
```

## Premier lancement Phase 2/3
- Téléchargement ModelNet40 (~500 MB)
- Voxelisation (quelques minutes, une seule fois)
- Ensuite le cache est réutilisé
