"""
Serveur FedAvg -- coordination par rounds d'entrainement local
================================================================
Alternative au systeme de taches par gradient (framework/coordinator.py) :
au lieu d'echanger un gradient par requete, chaque volontaire recoit une
portion fixe du dataset (un "shard"), entraine localement plusieurs epoques,
puis renvoie UN SEUL delta de poids par round. Le serveur moyenne les deltas
recus (ponderes par la taille de shard de chacun) et avance au round suivant.

Objectif : reduire drastiquement le nombre d'allers-retours reseau et
eliminer le probleme de peremption des gradients (chaque round est
synchrone par construction).

Lancement : python scripts/run_server_fedavg.py --phase 1 --rounds 10 --local-epochs 1 --shards 3
"""
from __future__ import annotations

import argparse
import hashlib
import threading
import time

import numpy as np
from flask import Flask, jsonify, request

from jobs.progressive.models_pytorch import CNN2D, params_to_vector, vector_to_params
from jobs.progressive.data_providers import CIFAR10Provider
from framework.compression import encode_vector, decode_vector, raw_size_bytes, compression_ratio

app = Flask(__name__)
lock = threading.Lock()


def stable_shard_id(client_id: str, n_shards: int) -> int:
    h = hashlib.sha256(client_id.encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, n_shards)


class FedAvgState:
    def __init__(self, n_shards: int, max_rounds: int, local_epochs: int,
                 batch_size: int, lr: float, target_accuracy: float,
                 round_timeout: float):
        self.n_shards = n_shards
        self.max_rounds = max_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.target_accuracy = target_accuracy
        self.round_timeout = round_timeout

        self.data = CIFAR10Provider()
        self.model = CNN2D(n_classes=10)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.theta = params_to_vector(self.model)

        self.round_id = 0
        self.round_start = time.time()
        self.submissions = {}          # client_id -> (delta_vec, n_samples)
        self.assigned_shards = {}      # client_id -> shard_id
        self.finished = False
        self.history = []
        self.start_time = time.time()

        self.bytes_uploaded_total = 0
        self.bytes_downloaded_total = 0

    def assign_shard(self, client_id: str) -> int:
        if client_id not in self.assigned_shards:
            self.assigned_shards[client_id] = stable_shard_id(client_id, self.n_shards)
        return self.assigned_shards[client_id]

    def maybe_aggregate(self):
        """Agrege si tous les shards ont soumis, ou si le timeout du round est depasse
        avec au moins une soumission. Doit etre appele avec le lock deja pris."""
        if self.finished:
            return
        n_sub = len(self.submissions)
        timed_out = (time.time() - self.round_start) > self.round_timeout
        if n_sub == 0:
            return
        if n_sub < self.n_shards and not timed_out:
            return

        # --- Agregation ponderee par le nombre d'echantillons locaux ---
        total_samples = sum(n for _, n in self.submissions.values())
        agg_delta = np.zeros_like(self.theta)
        for delta, n in self.submissions.values():
            agg_delta += delta * (n / total_samples)
        self.theta = self.theta + agg_delta

        # --- Evaluation ---
        vector_to_params(self.model, self.theta)
        import torch
        self.model.eval()
        x, y = self.data.sample_eval(1000)
        with torch.no_grad():
            xt = torch.from_numpy(x)
            pred = self.model(xt).argmax(dim=1).numpy()
        acc = float((pred == y).mean())

        elapsed = time.time() - self.start_time
        print(f"[FedAvg] round {self.round_id} termine "
              f"({n_sub}/{self.n_shards} volontaires, {'TIMEOUT' if timed_out else 'complet'}) "
              f"-> precision={acc*100:.2f}%  elapsed={elapsed:.0f}s")

        self.history.append({
            "round": self.round_id,
            "accuracy": acc,
            "participants": n_sub,
            "elapsed": elapsed,
        })

        self.round_id += 1
        self.submissions = {}
        self.round_start = time.time()

        if self.round_id >= self.max_rounds or acc >= self.target_accuracy:
            self.finished = True
            print(f"[FedAvg] TERMINE en {elapsed:.0f}s ({elapsed/60:.1f} min) "
                  f"-- precision finale={acc*100:.2f}%")


state: FedAvgState | None = None


@app.route("/fedavg/config")
def fedavg_config():
    return jsonify({
        "n_shards": state.n_shards,
        "n_params": state.n_params,
        "local_epochs": state.local_epochs,
        "batch_size": state.batch_size,
        "max_rounds": state.max_rounds,
        "target_accuracy": state.target_accuracy,
    })


@app.route("/fedavg/round")
def fedavg_round():
    client_id = request.args.get("client_id", "anon")
    with lock:
        state.maybe_aggregate()  # verifie aussi le timeout pendant que quelqu'un poll
        shard_id = state.assign_shard(client_id)
        payload, nbytes, _ = encode_vector(state.theta, dtype="fp16")
        state.bytes_downloaded_total += nbytes
        return jsonify({
            "round_id": state.round_id,
            "shard_id": shard_id,
            "n_shards": state.n_shards,
            "local_epochs": state.local_epochs,
            "batch_size": state.batch_size,
            "weights": payload,
            "finished": state.finished,
        })


@app.route("/fedavg/submit", methods=["POST"])
def fedavg_submit():
    d = request.get_json(force=True)
    client_id = d["client_id"]
    round_id = int(d["round_id"])
    n_samples = int(d["n_samples"])
    delta = decode_vector(d["delta"])
    nbytes = len(d["delta"])  # approx (base64), suffisant pour les metriques

    with lock:
        if state.finished:
            return jsonify({"status": "finished"})
        if round_id != state.round_id:
            # Soumission perimee (le round a deja avance) -- on l'ignore proprement
            return jsonify({"status": "stale", "current_round": state.round_id})
        state.submissions[client_id] = (delta, n_samples)
        state.bytes_uploaded_total += nbytes
        print(f"[FedAvg] reçu round {round_id} de {client_id} "
              f"({len(state.submissions)}/{state.n_shards} pour ce round)")
        state.maybe_aggregate()
        return jsonify({"status": "ok"})


@app.route("/fedavg/status")
def fedavg_status():
    with lock:
        return jsonify({
            "round": state.round_id,
            "max_rounds": state.max_rounds,
            "finished": state.finished,
            "history": state.history,
            "n_shards": state.n_shards,
            "submitted_this_round": len(state.submissions),
            "elapsed": time.time() - state.start_time,
            "bytes_uploaded_total": state.bytes_uploaded_total,
            "bytes_downloaded_total": state.bytes_downloaded_total,
            "raw_equivalent_bytes": raw_size_bytes(state.n_params) * 2
                                    * max(1, state.round_id) * state.n_shards,
        })


def main():
    global state
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, help="Seule la phase 1 (CIFAR-10) est geree par ce prototype.")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shards", type=int, default=3, help="Nombre de volontaires attendus (partitions du dataset).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-accuracy", type=float, default=0.75)
    ap.add_argument("--round-timeout", type=float, default=600.0,
                     help="Secondes avant d'agreger meme si tous les volontaires n'ont pas soumis.")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    state = FedAvgState(
        n_shards=args.shards, max_rounds=args.rounds, local_epochs=args.local_epochs,
        batch_size=args.batch_size, lr=args.lr, target_accuracy=args.target_accuracy,
        round_timeout=args.round_timeout,
    )

    print("================================================================")
    print("  SERVEUR FEDAVG -- rounds d'entrainement local")
    print("================================================================")
    print(f"  Modele        : CNN2D CIFAR-10 ({state.n_params:,} params)")
    print(f"  Rounds prevus : {args.rounds}")
    print(f"  Shards        : {args.shards} (= nombre de volontaires attendus)")
    print(f"  Epoques locales/round : {args.local_epochs}")
    print(f"  Port          : {args.port}")
    print("================================================================")
    print(f"  Commande volontaire : python scripts/run_volunteer_fedavg.py "
          f"--server http://<IP>:{args.port} --device <nom>")
    print("================================================================")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
