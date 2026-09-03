#!/usr/bin/env python3
"""
Reference SEQUENTIELLE (un seul appareil)
=========================================
Execute exactement le meme apprentissage par gradients, mais sur UN seul worker
en serie (aucun parallelisme). Sert d'ancre de comparaison pour demontrer la
plus-value du calcul volontaire distribue.

Enregistre, par epoque : precision, top-3, F1, tours moyens, nombre de pas de
gradient et temps ecoule -> results/sequential_run.json

  python scripts/run_sequential.py                 # jusqu'a la cible
  python scripts/run_sequential.py --epochs 18     # nb d'epoques fixe (courbes completes)
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT
from jobs.cnn_3d.attention_job import AttentionCNN3DJob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None,
                    help="nb d'epoques fixe (sinon, jusqu'a la cible)")
    args = ap.parse_args()

    cfg = DEFAULT
    job = AttentionCNN3DJob()
    max_epochs = args.epochs or cfg.train.max_epochs

    theta = job.init_params()
    opt = job.init_opt_state()
    eps = cfg.model.epsilon_start
    history = []
    step = 0
    t0 = time.time()
    grad_time_acc = 0.0  # temps CPU cumule de calcul des gradients (pour la modelisation)

    print("Reference sequentielle (1 appareil) :")
    for e in range(1, max_epochs + 1):
        for i in range(cfg.train.n_tasks_per_epoch):
            task = job.make_task(e, i, seed=step * 7919 + cfg.data.seed, epsilon=eps)
            tg = time.time()
            g, ns, lm = job.compute_gradient(theta, task)
            theta, opt = job.apply_gradient(theta, g, opt, cfg.model.lr)
            grad_time_acc += time.time() - tg
            step += 1
        m = job.evaluate(theta, cfg.server.validation_samples)
        history.append({"epoch": e, "accuracy": m["accuracy"], "top3": m["top3"],
                        "macro_f1": m["macro_f1"], "avg_turns": m["avg_turns"],
                        "grad_steps": step, "elapsed": time.time() - t0})
        print(f"  epoque {e:2d}  acc={m['accuracy']:.3f}  top3={m['top3']:.3f}  "
              f"F1={m['macro_f1']:.3f}  tours={m['avg_turns']:.1f}  "
              f"pas={step}  ({history[-1]['elapsed']:.1f}s)")
        eps = max(cfg.model.epsilon_end, eps * cfg.model.epsilon_decay)
        if args.epochs is None and m["accuracy"] >= cfg.train.target_accuracy:
            break

    out = {
        "history": history,
        "n_params": job.n_params(),
        "total_grad_steps": step,
        "total_time": time.time() - t0,
        "per_gradient_seconds": grad_time_acc / max(1, step),
        "target_accuracy": cfg.train.target_accuracy,
    }
    with open(os.path.join(RESULTS, "sequential_run.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTemps par gradient : {out['per_gradient_seconds']*1000:.1f} ms | "
          f"pas totaux : {step}")
    print("Metriques -> results/sequential_run.json")


if __name__ == "__main__":
    main()
