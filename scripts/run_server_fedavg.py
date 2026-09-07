#!/usr/bin/env python3
"""
Lancement du serveur FedAvg -- prototype "rounds d'entrainement local"
========================================================================
  python scripts/run_server_fedavg.py --rounds 10 --local-epochs 1 --shards 3

Alternative au serveur principal (scripts/run_server.py) : au lieu d'un
gradient par requete, chaque volontaire recoit un shard fixe et renvoie un
delta de poids par round complet d'entrainement local. Objectif : reduire
le nombre d'allers-retours reseau et le temps total, au prix d'un nombre
de volontaires connu a l'avance (--shards).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.fedavg_server import main

if __name__ == "__main__":
    main()
