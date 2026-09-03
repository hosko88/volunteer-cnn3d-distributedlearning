#!/usr/bin/env python3
"""
Preparation / verification des donnees (systeme progressif PyTorch)
=====================================================================
Declenche le telechargement reel des donnees pour la phase demandee.
A lancer avant une simulation ou un deploiement, SUR CHAQUE MACHINE
(serveur ET volontaires) : chacune doit disposer localement des donnees
pour calculer un gradient reel (principe du calcul volontaire : seuls les
parametres/gradients transitent sur le reseau, jamais les donnees brutes).

  python scripts/prepare_data.py --phase 1   # CIFAR-10 (torchvision, ~170 Mo)
  python scripts/prepare_data.py --phase 2   # ModelNet40 (point clouds -> voxels 32^3)
  python scripts/prepare_data.py --phase 3   # ModelNet40 (meme donnees que phase 2, + attention)
  python scripts/prepare_data.py --phase all

Note : le telechargement + la voxelisation de ModelNet40 (phases 2/3) sont
entierement automatiques (aucune etape manuelle), mais peuvent prendre
plusieurs minutes la premiere fois (conversion des nuages de points en
volumes 32x32x32, mise en cache ensuite).
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.progressive.data_providers import CIFAR10Provider, ModelNet40Provider


def prepare_phase1():
    print("\n=== Phase 1 - CIFAR-10 (CNN 2D) ===")
    provider = CIFAR10Provider(n_classes=10)
    provider._ensure()
    print(f"[OK] train={len(provider._train)}  test={len(provider._test)}")


def prepare_phase2_or_3(label):
    print(f"\n=== {label} - ModelNet40 (CNN 3D, voxels 32^3) ===")
    print("Telechargement + voxelisation automatiques (peut prendre plusieurs "
          "minutes la premiere fois, mise en cache ensuite) ...")
    provider = ModelNet40Provider(n_classes=40)
    provider._ensure()
    print(f"[OK] train={len(provider._train_x)}  test={len(provider._test_x)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "3", "all"], default="1")
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        prepare_phase1()
    if args.phase in ("2", "all"):
        prepare_phase2_or_3("Phase 2")
    if args.phase in ("3", "all"):
        prepare_phase2_or_3("Phase 3")

    print("\nPipeline pret. Lancement :")
    print("  python scripts/run_server.py --phase 1   (ou 2, ou 3)")
    print("  python scripts/run_volunteer.py --server http://<ip_serveur>:5000")


if __name__ == "__main__":
    main()
