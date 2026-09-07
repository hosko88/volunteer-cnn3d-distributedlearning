"""
Volontaire FedAvg -- entrainement local par round
===================================================
Contrepartie de framework/fedavg_server.py. Recoit les poids globaux et son
shard de donnees assigne, entraine localement plusieurs epoques, renvoie
UN SEUL delta de poids par round (au lieu d'un gradient par requete).

Lancement : python scripts/run_volunteer_fedavg.py --server http://<IP>:5001 --device pc1
"""
from __future__ import annotations

import argparse
import time
import uuid

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim

from jobs.progressive.models_pytorch import CNN2D, params_to_vector, vector_to_params
from jobs.progressive.data_providers import CIFAR10Provider
from framework.compression import encode_vector, decode_vector


def local_train(model, x_shard, y_shard, n_local_epochs, batch_size, lr, device):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(x_shard)
    for ep in range(n_local_epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) == 0:
                continue
            xb = torch.from_numpy(x_shard[idx]).to(device)
            yb = torch.from_numpy(y_shard[idx]).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, required=True)
    ap.add_argument("--device", type=str, default="pc")
    ap.add_argument("--poll-interval", type=float, default=2.0)
    args = ap.parse_args()

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_id = f"{args.device}-{uuid.uuid4().hex[:6]}"
    print(f"Client id : {client_id}")

    cfg = requests.get(f"{args.server}/fedavg/config", timeout=30).json()
    print(f"Config recue : {cfg}")

    provider = CIFAR10Provider()
    shard_id_cached = None
    x_shard = y_shard = None
    last_submitted_round = -1

    model = CNN2D(n_classes=10).to(torch_device)

    print("Connexion au serveur FedAvg, en attente de rounds...")
    while True:
        r = requests.get(f"{args.server}/fedavg/round",
                          params={"client_id": client_id}, timeout=30).json()

        if r["finished"]:
            print("Entrainement termine cote serveur. Arret du volontaire.")
            break

        round_id = r["round_id"]
        shard_id = r["shard_id"]
        n_shards = r["n_shards"]

        if round_id == last_submitted_round:
            # Deja soumis pour ce round, on attend que les autres terminent
            time.sleep(args.poll_interval)
            continue

        # Charge le shard une seule fois (ou si l'assignation change, en theorie stable)
        if shard_id_cached != shard_id:
            print(f"Chargement du shard {shard_id}/{n_shards}...")
            x_shard, y_shard = provider.get_shard(shard_id, n_shards)
            shard_id_cached = shard_id
            print(f"  -> {len(x_shard)} images locales")

        # Charge les poids globaux du round courant
        theta = decode_vector(r["weights"])
        vector_to_params(model, theta)

        t0 = time.time()
        local_train(model, x_shard, y_shard,
                    n_local_epochs=r["local_epochs"],
                    batch_size=r["batch_size"],
                    lr=1e-3, device=torch_device)
        dt = time.time() - t0

        new_theta = params_to_vector(model)
        delta = new_theta - theta
        payload, nbytes, _ = encode_vector(delta, dtype="fp16")

        resp = requests.post(f"{args.server}/fedavg/submit", json={
            "client_id": client_id,
            "round_id": round_id,
            "n_samples": len(x_shard),
            "delta": payload,
        }, timeout=30).json()

        print(f"Round {round_id} termine en {dt:.1f}s "
              f"({len(x_shard)} images, {r['local_epochs']} epoque(s) locale(s)) "
              f"-> {resp.get('status')}")
        last_submitted_round = round_id


if __name__ == "__main__":
    main()
