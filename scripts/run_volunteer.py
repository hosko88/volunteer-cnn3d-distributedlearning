#!/usr/bin/env python3
"""
Lancement d'un volontaire (smartphone / PC Linux / PC Windows)
=============================================================
  python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000
  python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device smartphone

Options :
  --device   etiquette affichee (ex: smartphone, linux-pc, windows-pc)
  --power    nb de sous-taches par requete (par defaut : auto selon l'appareil)
  --slowdown ralentissement artificiel par sous-tache (s), pour simuler un appareil lent
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.volunteer import VolunteerClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="ex: http://192.168.1.10:5000")
    ap.add_argument("--device", default=None)
    ap.add_argument("--power", type=int, default=None)
    ap.add_argument("--slowdown", type=float, default=0.0)
    args = ap.parse_args()

    client = VolunteerClient(args.server, device_label=args.device,
                             power=args.power, slowdown=args.slowdown)
    client.run()


if __name__ == "__main__":
    main()
