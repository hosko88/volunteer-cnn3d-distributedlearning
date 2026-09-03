"""
Client volontaire
=================
Tourne sur un smartphone (Termux/Pydroid), un PC Linux ou Windows. Boucle :
  1. telecharge les parametres globaux theta depuis le serveur ;
  2. recoit un lot de sous-taches (proportionnel a sa puissance),
     chaque sous-tache contenant UNIQUEMENT le mini-batch nécessaire ;
  3. pour chaque sous-tache, calcule le gradient sur les données reçues,
     puis le compresse ;
  4. renvoie les gradients compressés au serveur.

Le volontaire ne télécharge JAMAIS le dataset complet ni une partition.
Le serveur est le seul à posséder CIFAR-10 / ModelNet40 et n'envoie que
de petits mini-batches (float16 + npz) dans chaque tâche.

Robustesse : en cas d'erreur reseau, on reessaie ; les sous-taches perdues
sont reattribuees par le serveur (le volontaire n'a rien de special a faire).
"""

import time
import io
import base64
import requests
import numpy as np

from config import DEFAULT
from framework.compression import encode_vector, decode_vector
from client.device_info import get_device_info, estimate_power, benchmark_2s


def _detect_torch() -> bool:
    """Vrai si PyTorch est utilisable sur cette machine (permet au serveur de
    router ce volontaire vers la piste lourde PyTorch ou la piste legere
    NumPy dans un pool heterogene)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def build_job_from_spec(spec, cfg):
    """Reconstruit le job a partir de la spec envoyee par le serveur."""
    job_type = spec.get("job_type", "")
    kb = spec.get("kb", {})

    # Jobs progressifs (PyTorch) -- Phase 1 / 2 / 3. Import differe (pas en
    # tete de fonction) : un volontaire sans PyTorch (ex. smartphone Termux)
    # ne doit planter QUE s'il recoit reellement une tache PyTorch, pas a
    # chaque appel de cette fonction.
    if job_type in ("Phase1CIFAR2DJob", "Phase2ModelNet3DJob", "Phase3Attention3DJob") \
            or kb.get("kind") in ("cifar2d", "modelnet3d", "attention3d") \
            or kb.get("phase") in (1, 2, 3):
        from jobs.progressive import (
            Phase1CIFAR2DJob, Phase2ModelNet3DJob, Phase3Attention3DJob,
        )
        if job_type == "Phase1CIFAR2DJob" or kb.get("kind") == "cifar2d" or kb.get("phase") == 1:
            return Phase1CIFAR2DJob.from_kb_spec(kb, cfg)
        if job_type == "Phase2ModelNet3DJob" or kb.get("kind") == "modelnet3d" or kb.get("phase") == 2:
            return Phase2ModelNet3DJob.from_kb_spec(kb, cfg)
        if job_type == "Phase3Attention3DJob" or kb.get("kind") == "attention3d" or kb.get("phase") == 3:
            return Phase3Attention3DJob.from_kb_spec(kb, cfg)

    # Compatibilite anciens jobs NumPy (aucune dependance a torch ici)
    try:
        from jobs.cnn_3d.attention_job import AttentionCNN3DJob
        return AttentionCNN3DJob.from_kb_spec(kb, cfg)
    except Exception:
        from jobs.progressive import Phase1CIFAR2DJob
        return Phase1CIFAR2DJob.from_kb_spec(kb, cfg)


class VolunteerClient:
    def __init__(self, server_url, device_label=None, power=None,
                 slowdown=0.0, max_iters=100000, cfg=DEFAULT):
        self.server = server_url.rstrip("/")
        self.cfg = cfg
        self.slowdown = slowdown
        self.max_iters = max_iters
        self.info = get_device_info(device_label)
        self.info["has_torch"] = _detect_torch()

        print("Benchmark local de 2 secondes en cours...")
        benchmark_score = benchmark_2s(2.0)
        self.info["benchmark_score"] = benchmark_score

        self.power = power if power is not None else estimate_power(self.info)

        print(f"Benchmark terminé : score={benchmark_score}, puissance={self.power}")
        self.client_id = f"{self.info['device']}-{int(time.time()*1000) % 100000}"
        self.job = None
        self.tcfg = cfg.transport
        # Error Feedback residual (améliore la convergence avec compression agressive)
        self._grad_residual = None

    def _get_job(self):
        for _ in range(30):
            try:
                r = requests.get(
                    f"{self.server}/kb",
                    params={"has_torch": "1" if self.info.get("has_torch") else "0"},
                    timeout=10,
                ).json()
                self.job = build_job_from_spec(r, self.cfg)
                self.tcfg.dtype = r["transport"]["dtype"]
                self.tcfg.topk = r["transport"]["topk"]
                self._track = r.get("track", "heavy")
                print(f"Piste assignee : {self._track} "
                      f"({'PyTorch' if self.info.get('has_torch') else 'NumPy leger'})")
                # Plus de téléchargement de shard/partition.
                # Les mini-batches sont inclus dans chaque tâche par le serveur.
                return
            except requests.RequestException:
                time.sleep(1.0)
        raise RuntimeError("serveur injoignable")

    def wait_for_server(self):
        """
        Attend que le serveur autorise le démarrage.
        """

        while True:
            try:
                r = requests.get(
                    f"{self.server}/training_status",
                    timeout=10
                ).json()

                if r["started"]:
                    print("\n>>> Départ reçu du serveur.")
                    return

                print("En attente du signal du serveur...")

            except requests.RequestException:
                pass

            time.sleep(2)    
            
    def run(self, verbose=True):
        self._get_job()
        print("\nConnexion réussie.")
        print("Le volontaire attend le signal de départ...")
        self.wait_for_server()

        if verbose:
            print(f"[volontaire {self.client_id}] {self.info['os']} "
                  f"cpu={self.info['cpu']} puissance={self.power} -> {self.server}")
        it = 0
        while it < self.max_iters:
            it += 1
            try:
                request_start = time.time()

                response = requests.post(f"{self.server}/request_work", timeout=15, json={
                    "client_id": self.client_id,
                    "info": self.info,
                    "power": self.power
                })

                request_work_seconds = time.time() - request_start
                resp = response.json()

            except requests.RequestException:
                time.sleep(0.5)
                continue

            if resp.get("finished"):
                if verbose:
                    print(f"[volontaire {self.client_id}] calcul termine, arret.")
                return
            tasks = resp.get("tasks", [])
            if not tasks:
                time.sleep(0.2); continue

            theta = decode_vector(resp["params"])
            version = resp["params_version"]
            results = []
            for task in tasks:
                task_start = time.time()

                grad, n_samples, lm = self.job.compute_gradient(theta, task)

                task_duration = time.time() - task_start

                payload, _, self._grad_residual = encode_vector(
                    grad, dtype=self.tcfg.dtype, topk=self.tcfg.topk,
                    residual=self._grad_residual
                )

                lm["duration_seconds"] = task_duration
                lm["request_work_seconds"] = request_work_seconds / max(1, len(tasks))
                lm["client_device"] = self.info.get("device")
                lm["client_os"] = self.info.get("os")
                lm["client_cpu"] = self.info.get("cpu")
                lm["client_ram_gb"] = self.info.get("ram_gb")

                results.append({
                    "task_id": task["task_id"],
                    "grad": payload,
                    "params_version": version,
                    "n_samples": n_samples,
                    "local_metrics": lm
                })
                if self.slowdown:
                    time.sleep(self.slowdown)
            try:
                report_start = time.time()

                requests.post(
                    f"{self.server}/report",
                    timeout=15,
                    json={"client_id": self.client_id, "results": results}
                )

                report_seconds = time.time() - report_start

                requests.post(
                    f"{self.server}/client_comm_metrics",
                    timeout=10,
                    json={
                        "client_id": self.client_id,
                        "task_ids": [r["task_id"] for r in results],
                        "report_seconds": report_seconds,
                        "tasks_count": len(results)
                    }
                )

                if verbose:
                    print(
                        f"[communication] request_work={request_work_seconds:.4f}s "
                        f"report={report_seconds:.4f}s "
                        f"tasks={len(results)}"
                    )

            except requests.RequestException:
                pass
        if verbose:
            print(f"[volontaire {self.client_id}] limite d'iterations atteinte.")