#!/usr/bin/env python3
"""
Lancement d'un volontaire FedAvg -- prototype "rounds d'entrainement local"
==============================================================================
  python scripts/run_volunteer_fedavg.py --server http://<IP>:5001 --device pc1

Contrepartie de scripts/run_server_fedavg.py. Necessite PyTorch (piste
lourde uniquement -- pas de version legere pour ce prototype pour l'instant).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.fedavg_volunteer import main

if __name__ == "__main__":
    main()
