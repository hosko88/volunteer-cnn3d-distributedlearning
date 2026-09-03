#!/usr/bin/env python3
"""
Lancement du serveur de coordination — Progression 2D → 3D + pool heterogene
=============================================================================
  python scripts/run_server.py --phase 1          # CIFAR-10 / CNN 2D
  python scripts/run_server.py --phase 2          # ModelNet40 / CNN 3D
  python scripts/run_server.py --phase 3          # ModelNet40 / CNN 3D + Attention
  python scripts/run_server.py --host 0.0.0.0 --phase 1
  python scripts/run_server.py --phase 1 --no-light-track   # desactive la piste NumPy/smartphone

Pool heterogene (active par defaut) : en plus du job PyTorch principal
(piste "heavy", PC/laptops), un second job NumPy leger (piste "light",
smartphones/appareils sans PyTorch) tourne en parallele sur le meme serveur.
Chaque volontaire est route automatiquement vers la piste adaptee a ses
capacites declarees.
"""

import os
import sys
import socket
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT
from jobs.progressive import Phase1CIFAR2DJob, Phase2ModelNet3DJob, Phase3Attention3DJob
from jobs.cnn_3d.attention_job import AttentionCNN3DJob
from server.app import create_app


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def build_job(phase: int):
    if phase == 1:
        return Phase1CIFAR2DJob()
    if phase == 2:
        return Phase2ModelNet3DJob()
    if phase == 3:
        return Phase3Attention3DJob()
    raise ValueError(f"Phase inconnue: {phase}")


def banner(host, port, job_heavy, phase, job_light):
    ip = local_ip() if host == "0.0.0.0" else host
    line = "=" * 64
    print(line)
    print("  SERVEUR DE CALCUL VOLONTAIRE  —  PROGRESSION 2D → 3D")
    print(line)
    print(f"  Phase                    : {phase}")
    print(f"  Piste lourde (PyTorch)   : {job_heavy.name}  ({job_heavy.n_params():,} params)")
    if job_light is not None:
        print(f"  Piste legere (NumPy)     : {job_light.name}  ({job_light.n_params():,} params)")
    else:
        print("  Piste legere             : desactivee (--no-light-track)")
    print(f"  Adresse                  : http://{ip}:{port}")
    print(f"  Tableau de bord          : http://{ip}:{port}/")
    print(line)
    print("  Commandes volontaire :")
    print(f"    PC/laptop  (PyTorch)  : python scripts/run_volunteer.py --server http://{ip}:{port}")
    print(f"    Smartphone (Termux)   : identique -- la piste est choisie automatiquement")
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT.server.port)
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], default=1,
                    help="1=CIFAR2D, 2=ModelNet3D, 3=Attention3D (piste lourde uniquement)")
    ap.add_argument("--auto-start", action="store_true")
    ap.add_argument("--no-light-track", action="store_true",
                    help="desactive la piste NumPy legere (smartphones non acceptes)")
    args = ap.parse_args()

    job_heavy = build_job(args.phase)
    job_light = None if args.no_light_track else AttentionCNN3DJob()

    banner(args.host, args.port, job_heavy, args.phase, job_light)

    app = create_app(job_heavy, DEFAULT, job_light=job_light, auto_start=args.auto_start)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
