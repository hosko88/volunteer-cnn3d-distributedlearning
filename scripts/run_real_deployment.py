#!/usr/bin/env python3
"""Script de démarrage rapide pour un déploiement réel entre machines."""

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT
from jobs.cnn_3d.job import CNN3DJob
from jobs.cnn_3d.attention_job import AttentionCNN3DJob
from server.app import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["server", "volunteer"], required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--server", default=None, help="URL du serveur pour un volontaire")
    ap.add_argument("--device", default=None)
    ap.add_argument("--power", type=int, default=None)
    ap.add_argument("--slowdown", type=float, default=0.0)
    ap.add_argument(
        "--job",
        choices=["cnn3d", "attention"],
        default="attention",
        help="job principal de démonstration : attention (CNN 3D)"
    )
    args = ap.parse_args()

    if args.mode == "server":
        if args.job == "cnn3d":
            job = CNN3DJob()
        else:
            job = AttentionCNN3DJob()
        app = create_app(job, DEFAULT)
        app.coord.start()
        print(textwrap.dedent(f"""
        Serveur prêt.
        Adresse : http://{args.host}:{args.port}
        Tableau de bord : http://{args.host}:{args.port}/
        Lancez ensuite un volontaire avec :
        python scripts/run_real_deployment.py --mode volunteer --server http://{args.host}:{args.port} --device laptop
        """))
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    else:
        from client.volunteer import VolunteerClient
        client = VolunteerClient(args.server, device_label=args.device,
                                 power=args.power, slowdown=args.slowdown)
        client.run(verbose=True)


if __name__ == "__main__":
    main()
