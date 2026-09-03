#!/usr/bin/env python3
"""
Simulation distribuee de bout en bout (sur une seule machine)
============================================================
Lance le serveur puis une FLOTTE HETEROGENE de volontaires (PC Windows, PC Linux,
smartphones) en fils d'execution, attend la convergence, puis enregistre toutes
les metriques utiles au memoire dans results/distributed_run.json.

  python scripts/run_simulation.py
  python scripts/run_simulation.py --kill      # tue un smartphone en cours (tolerance aux pannes)
  python scripts/run_simulation.py --topk 0.25 # transport encore plus leger (sparsification)

NOTE : cette machine de demonstration peut n'avoir qu'UN coeur ; les fils se
partagent alors le CPU et le gain de temps PARALLELE n'est pas observable ici
(il l'est sur un vrai deploiement multi-appareils). La simulation valide la
CORRECTION, la CONVERGENCE, la TOLERANCE AUX PANNES, l'HETEROGENEITE et le
transport par gradients. Le gain de temps est quantifie par scripts/analyze.py.
"""

import os
import sys
import json
import time
import argparse
import threading
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from werkzeug.serving import make_server

from config import DEFAULT
from jobs.cnn_3d.attention_job import AttentionCNN3DJob
from server.app import create_app
from client.volunteer import VolunteerClient

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# flotte heterogene : (etiquette, puissance, ralentissement_s_par_sous-tache)
FLEET = [
    ("windows-pc", 3, 0.00),
    ("linux-pc",   2, 0.01),
    ("smartphone-1", 1, 0.05),
    ("smartphone-2", 1, 0.08),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--kill", action="store_true", help="tue smartphone-2 en cours de route")
    ap.add_argument("--topk", type=float, default=None, help="sparsification des gradients (<1)")
    ap.add_argument("--dtype", default=None, choices=["fp16", "fp32"])
    args = ap.parse_args()

    if args.topk is not None:
        DEFAULT.transport.topk = args.topk
    if args.dtype:
        DEFAULT.transport.dtype = args.dtype

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    job = AttentionCNN3DJob()
    app = create_app(job, DEFAULT)
    app.coord.start()

    srv = make_server("127.0.0.1", args.port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{args.port}"

    # attendre que le serveur reponde
    for _ in range(50):
        try:
            requests.get(f"{base}/status", timeout=2); break
        except requests.RequestException:
            time.sleep(0.1)

    print(f"Simulation distribuee : {len(FLEET)} volontaires heterogenes, "
          f"transport gradients {DEFAULT.transport.dtype} top-k={DEFAULT.transport.topk}")
    print(f"Tableau de bord en direct : {base}/\n")

    threads = []
    for label, power, slow in FLEET:
        max_iters = 3 if (args.kill and label == "smartphone-2") else 100000
        c = VolunteerClient(base, device_label=label, power=power,
                            slowdown=slow, max_iters=max_iters)
        t = threading.Thread(target=c.run, kwargs={"verbose": False}, daemon=True)
        t.start(); threads.append(t)
        if args.kill and label == "smartphone-2":
            print("  [!] smartphone-2 ne fera que 3 lots puis se deconnecte "
                  "(test de tolerance aux pannes)")

    # suivi en direct dans la console
    last_epoch = 0
    while not app.coord.finished:
        st = app.coord.status()
        if st["epoch"] != last_epoch and st["history"]:
            h = st["history"][-1]
            print(f"  epoque {h['epoch']:2d}  acc={h['accuracy']:.3f}  "
                  f"top3={h.get('top3', 0):.3f}  tours={h.get('avg_turns', 0):.1f}  "
                  f"reattributions={h['reassigned']}  v={h['version']}  "
                  f"({h['elapsed']:.1f}s)")
            last_epoch = st["epoch"]
        time.sleep(0.3)

    # rapport final
    st = app.coord.status()
    h = st["history"][-1]
    bw = st["bandwidth"]
    print("\n=== CALCUL TERMINE ===")
    print(f"Precision finale : {h['accuracy']:.3f} | top-3 {h.get('top3',0):.3f} | "
          f"F1 {h.get('macro_f1',0):.3f} | tours {h.get('avg_turns',0):.1f}")
    print(f"Epoques : {h['epoch']} | temps : {h['elapsed']:.1f}s | "
          f"gradients appliques : {bw['gradients_appliques']} "
          f"(perimes rejetes : {bw['gradients_perimes_rejetes']})")
    print(f"Reattributions (tolerance pannes) : {st['scheduler']['reassigned']}")
    print(f"Bande passante : {bw['octets_total_compresse']/1e6:.2f} Mo compresses "
          f"vs {bw['octets_equivalent_brut_fp32']/1e6:.2f} Mo bruts fp32 "
          f"-> reduction x{bw['facteur_reduction']:.1f}")
    print("\nContribution par appareil :")
    for cid, c in st["scheduler"]["clients"].items():
        print(f"  {c['info']['device']:14s} lots={c['completed']:4d}  "
              f"fiabilite={c['reliability']:.2f}  puissance={c['power']}")

    out = {
        "history": st["history"],
        "bandwidth": bw,
        "scheduler": st["scheduler"],
        "fleet": FLEET,
        "transport": {"dtype": DEFAULT.transport.dtype, "topk": DEFAULT.transport.topk},
        "wall_time": h["elapsed"],
        "n_params": job.n_params(),
    }
    with open(os.path.join(RESULTS, "distributed_run.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nMetriques -> results/distributed_run.json")
    srv.shutdown()


if __name__ == "__main__":
    main()
