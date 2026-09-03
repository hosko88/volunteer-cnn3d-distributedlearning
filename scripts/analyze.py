#!/usr/bin/env python3
"""
Analyse et metriques pour le memoire
====================================
Produit, a partir des executions sauvegardees (et de quelques mesures), les
figures et chiffres qui DEMONTRENT LA THESE : le calcul volontaire d'appareils
grand public heterogenes peut realiser l'apprentissage distribue sur un job CNN 3D
progressif, sans super-calculateur, a cout quasi nul, et plus vite qu'un systeme
sequentiel.

Figures produites dans results/ :
  fig1_convergence_qualite.png   qualite de la phase (precision/top-3/F1)
  fig2_plus_value.png            plus-value vs sequentiel (precision vs temps, K appareils)
  fig3_passage_echelle.png       temps-pour-la-cible vs nombre de volontaires
  fig4_bande_passante.png        allegement du transport (gradients compresses vs poids bruts)
  fig5_heterogeneite.png         repartition de la charge selon la puissance
  fig6_tolerance_pannes.png      convergence malgre la perte de volontaires
  fig7_cout_accessibilite.png    cout et energie : flotte vs GPU cloud vs serveur

  + results/metrics_memoire.json : tous les chiffres regroupes.

NOTE D'HONNETETE : cette machine peut n'avoir qu'un coeur. Le gain de TEMPS
parallele n'y est donc pas mesurable directement ; il est MODELISE a partir du
temps par gradient reellement mesure (et se materialise sur un vrai deploiement
multi-appareils). Tout le reste (convergence, qualite, tolerance aux pannes,
heterogeneite, bande passante) est MESURE.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT
from framework.compression import encode_vector, raw_size_bytes
from jobs.cnn_3d.attention_job import AttentionCNN3DJob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

C_FLEET, C_CLOUD, C_SERVER, C_SEQ = "#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True})


# ===== hypotheses du modele de cout (EDITABLES -- a remplacer par vos chiffres) ===== #
ASSUMPTIONS = {
    "prix_electricite_usd_kwh": 0.15,        # ordre de grandeur Afrique subsaharienne
    "puissance_smartphone_w": 4.0,           # smartphone en calcul
    "puissance_pc_w": 65.0,                  # PC portable/bureau
    "fraction_marginale_volontaire": 0.6,    # surcout marginal (appareils deja allumes)
    "prix_gpu_cloud_usd_h": 1.20,            # location instance GPU
    "puissance_gpu_cloud_w": 300.0,
    "acceleration_gpu_vs_pc": 25.0,          # un GPU ~ 25x un PC volontaire sur cette charge
    "cout_amorti_serveur_usd_h": 6.0,        # serveur dedie : capital amorti
    "puissance_serveur_w": 2000.0,
    "acceleration_serveur_vs_pc": 60.0,
    "mix_flotte_smartphone": 0.75,           # une flotte africaine = surtout des smartphones
    "station_reference_en_pc": 12.0,         # station de travail ~ 12 PC volontaires
    "entrainement_realiste_heures_pc": 50.0, # echelle d'un entrainement reel (heures-PC) -- editable
}


def _load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ensure_sequential():
    data = _load("sequential_run.json")
    if data:
        return data
    print("  (sequential_run.json absent -> execution d'une reference sequentielle, ~30s)")
    os.system(f"{sys.executable} {os.path.join(HERE,'scripts','run_sequential.py')} --epochs 16")
    return _load("sequential_run.json")


# --------------------------------------------------------------------------- #
#  FIG 1 -- qualite medicale (depuis la reference sequentielle, courbes completes)
# --------------------------------------------------------------------------- #
def fig1_qualite(seq):
    h = seq["history"]
    ep = [r["epoch"] for r in h]
    acc = [r["accuracy"] for r in h]
    top3 = [r["top3"] for r in h]
    f1 = [r["macro_f1"] for r in h]
    turns = [r["avg_turns"] for r in h]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, acc, "-o", ms=3, color=C_FLEET, label="precision")
    ax1.plot(ep, top3, "-s", ms=3, color="#1f77b4", label="precision top-3")
    ax1.plot(ep, f1, "-^", ms=3, color="#9467bd", label="F1 macro")
    ax1.set_xlabel("Epoque"); ax1.set_ylabel("Qualite de la phase"); ax1.set_ylim(0, 1.02)
    ax1.legend(loc="center right")
    ax2 = ax1.twinx()
    ax2.plot(ep, turns, "--D", ms=3, color="#d62728", label="étapes de travail")
    ax2.set_ylabel("Étapes de travail", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728"); ax2.grid(False)
    ax2.set_ylim(0, max(1, DEFAULT.train.n_tasks_per_epoch))
    ax1.set_title("Qualite de la phase\n"
                  "(la precision monte ; les besoins de travail deviennent plus stables)")
    fig.savefig(os.path.join(RESULTS, "fig1_convergence_qualite.png"), dpi=130)
    plt.close(fig)
    return {"precision_finale": acc[-1], "top3_finale": top3[-1],
            "f1_finale": f1[-1], "tours_initial": turns[0], "tours_final": turns[-1]}


# --------------------------------------------------------------------------- #
#  FIG 2 -- plus-value vs sequentiel : precision vs TEMPS pour K appareils
# --------------------------------------------------------------------------- #
def fig2_plus_value(seq, Ks=(1, 4, 10)):
    h = seq["history"]
    steps = np.array([r["grad_steps"] for r in h])
    acc = np.array([r["accuracy"] for r in h])
    tg = seq["per_gradient_seconds"]
    target = seq["target_accuracy"]

    fig, ax = plt.subplots(figsize=(9, 5))
    cols = {1: C_SEQ, 4: C_FLEET, 10: "#1f77b4"}
    ttt = {}
    for K in Ks:
        temps = steps * tg / K   # temps mur modelise pour K appareils en parallele
        lbl = "sequentiel (1 appareil)" if K == 1 else f"distribue ({K} volontaires)"
        ax.plot(temps, acc, "-o", ms=3, color=cols.get(K, None), label=lbl)
        idx = np.argmax(acc >= target)
        if acc[idx] >= target:
            ttt[K] = float(temps[idx])
    ax.axhline(target, ls="--", color="#333", alpha=0.6, label=f"cible {target:.0%}")
    if 1 in ttt and ttt.get(max(Ks)):
        ax.annotate(f"cible atteinte ~{ttt[1]:.0f}s",
                    xy=(ttt[1], target), xytext=(ttt[1]*0.5, target-0.15),
                    fontsize=9, color=C_SEQ,
                    arrowprops=dict(arrowstyle="->", color=C_SEQ))
        Kmax = max(Ks)
        ax.annotate(f"~{ttt[Kmax]:.0f}s avec {Kmax} volontaires",
                    xy=(ttt[Kmax], target), xytext=(ttt[Kmax], target-0.32),
                    fontsize=9, color="#1f77b4",
                    arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    ax.set_xlabel("Temps mur estime (s)")
    ax.set_ylabel("Précision de la phase")
    ax.set_ylim(0, 1.02)
    ax.set_title("Plus-value du calcul volontaire vs systeme sequentiel\n"
                 "meme apprentissage, atteint la cible ~K fois plus vite avec K volontaires")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(RESULTS, "fig2_plus_value.png"), dpi=130)
    plt.close(fig)
    speedup = (ttt[1] / ttt[max(Ks)]) if (1 in ttt and max(Ks) in ttt) else None
    return {"temps_cible_par_K": ttt, "acceleration_K_max": speedup}


# --------------------------------------------------------------------------- #
#  FIG 3 -- passage a l'echelle : temps-pour-la-cible vs nombre de volontaires
# --------------------------------------------------------------------------- #
def fig3_echelle(seq, max_dev=60):
    steps_to_target = None
    for r in seq["history"]:
        if r["accuracy"] >= seq["target_accuracy"]:
            steps_to_target = r["grad_steps"]; break
    if steps_to_target is None:
        steps_to_target = seq["total_grad_steps"]
    tg = seq["per_gradient_seconds"]
    work_time = steps_to_target * tg            # temps CPU total pour atteindre la cible

    ns = np.arange(1, max_dev + 1)
    temps = work_time / ns                      # parfaitement parallele (borne haute)
    ref_pc = ASSUMPTIONS["station_reference_en_pc"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, temps, color=C_FLEET, lw=2)
    ax.axvline(ref_pc, ls="--", color=C_SERVER, alpha=0.7)
    ax.annotate(f"~{ref_pc:.0f} appareils\n= 1 station de travail",
                xy=(ref_pc, work_time/ref_pc), xytext=(ref_pc+4, work_time/ref_pc*1.6),
                fontsize=9)
    for k in (1, 4, 10, 30):
        if k <= max_dev:
            ax.annotate(f"{temps[k-1]:.0f}s", xy=(k, temps[k-1]), fontsize=8, color="#333")
    ax.set_xlabel("Nombre de volontaires")
    ax.set_ylabel("Temps estime pour atteindre la cible (s)")
    ax.set_title("Passage a l'echelle : ajouter des volontaires (peu couteux)\n"
                 "reduit le temps pour atteindre la precision cible")
    fig.savefig(os.path.join(RESULTS, "fig3_passage_echelle.png"), dpi=130)
    plt.close(fig)
    return {"pas_pour_cible": int(steps_to_target),
            "temps_1_appareil_s": float(work_time),
            "temps_30_appareils_s": float(work_time / 30)}


# --------------------------------------------------------------------------- #
#  FIG 4 -- allegement du transport : gradients compresses vs poids bruts
# --------------------------------------------------------------------------- #
def fig4_bande_passante(job):
    # un gradient representatif (sur un theta initial)
    theta = job.init_params()
    rng = np.random.default_rng(0)
    task = job.make_task(1, 0, seed=1, epsilon=0.3)
    grad, _, _ = job.compute_gradient(theta, task)
    n = job.n_params()

    schemes = []
    raw = raw_size_bytes(n, "fp32")
    schemes.append(("poids bruts fp32\n(approche naive)", raw, C_SERVER))
    _, b16, _ = encode_vector(grad, dtype="fp16", topk=1.0)
    schemes.append(("gradients fp16\n+ compression", b16, C_CLOUD))
    _, b16_25, _ = encode_vector(grad, dtype="fp16", topk=0.25)
    schemes.append(("gradients fp16\n+ top-25%", b16_25, C_FLEET))
    _, b16_10, _ = encode_vector(grad, dtype="fp16", topk=0.10)
    schemes.append(("gradients fp16\n+ top-10%", b16_10, "#17becf"))

    labels = [s[0] for s in schemes]
    sizes = [s[1] / 1024 for s in schemes]   # Ko
    cols = [s[2] for s in schemes]

    fig, ax = plt.subplots(figsize=(9, 5))
    b = ax.bar(labels, sizes, color=cols)
    for rect, sz in zip(b, sizes):
        ax.text(rect.get_x()+rect.get_width()/2, sz, f"{sz:.0f} Ko\n(x{raw/1024/sz:.1f})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Taille transmise par mise a jour (Ko)")
    ax.set_title("Allegement des echanges volontaire <-> serveur\n"
                 "envoyer des gradients compresses au lieu des poids reduit fortement le transfert")
    fig.savefig(os.path.join(RESULTS, "fig4_bande_passante.png"), dpi=130)
    plt.close(fig)
    return {"octets_poids_fp32": raw, "octets_grad_fp16": b16,
            "octets_grad_fp16_top25": b16_25, "octets_grad_fp16_top10": b16_10,
            "reduction_fp16": raw/b16, "reduction_top10": raw/b16_10}


# --------------------------------------------------------------------------- #
#  FIG 5 -- heterogeneite : repartition de la charge selon la puissance
# --------------------------------------------------------------------------- #
def fig5_heterogeneite(dist):
    if not dist:
        return None
    clients = dist["scheduler"]["clients"]
    items = sorted(clients.values(), key=lambda c: -c["completed"])
    names = [c["info"]["device"] for c in items]
    done = [c["completed"] for c in items]
    powers = [c["power"] for c in items]
    total = sum(done) or 1

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    part_reelle = [d/total for d in done]
    part_puissance = [p/sum(powers) for p in powers]
    w = 0.38
    ax.bar(x - w/2, part_reelle, w, color=C_FLEET, label="part reelle du travail")
    ax.bar(x + w/2, part_puissance, w, color="#1f77b4", alpha=0.8,
           label="part attendue (proportionnelle a la puissance)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Part du travail total")
    ax.set_title("L'ordonnanceur repartit la charge selon la puissance de chaque appareil\n"
                 "(le plus lent contribue sans bloquer : asynchrone + finalisation a "
                 f"{int(DEFAULT.train.completion_fraction*100)}%)")
    ax.legend()
    fig.savefig(os.path.join(RESULTS, "fig5_heterogeneite.png"), dpi=130)
    plt.close(fig)
    return {"contribution": {n: d for n, d in zip(names, done)},
            "reassignations": dist["scheduler"]["reassigned"]}


# --------------------------------------------------------------------------- #
#  FIG 6 -- tolerance aux pannes : convergence malgre la perte de volontaires
# --------------------------------------------------------------------------- #
def fig6_tolerance(job, cfg, drop_levels=(0.0, 0.2, 0.4, 0.6), max_epochs=12):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(drop_levels)))
    resume = {}
    for p, col in zip(drop_levels, colors):
        rng = np.random.default_rng(3)
        theta = job.init_params()
        opt = job.init_opt_state()
        eps = cfg.model.epsilon_start
        ep_acc = []
        for e in range(1, max_epochs + 1):
            for i in range(cfg.train.n_tasks_per_epoch):
                task = job.make_task(e, i, seed=int(rng.integers(1_000_000_000)), epsilon=eps)
                g, ns, lm = job.compute_gradient(theta, task)
                if rng.random() < p:
                    continue
                theta, opt = job.apply_gradient(theta, g, opt, cfg.model.lr)
            m = job.evaluate(theta, 300)
            ep_acc.append(m["accuracy"])
            eps = max(cfg.model.epsilon_end, eps * cfg.model.epsilon_decay)
        ax.plot(range(1, max_epochs+1), ep_acc, "-o", ms=3, color=col, lw=2,
                label=f"{p:.0%} de pertes (final {ep_acc[-1]:.2f})")
        resume[f"{p:.0%}"] = ep_acc[-1]
    ax.axhline(cfg.train.target_accuracy, ls="--", color="#333", alpha=0.6)
    ax.set_xlabel("Epoque"); ax.set_ylabel("Precision (validation)"); ax.set_ylim(0, 1.02)
    ax.set_title("Tolerance aux pannes : convergence malgre la deconnexion de volontaires\n"
                 "(batterie, reseau intermittent -- realite des smartphones)")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(RESULTS, "fig6_tolerance_pannes.png"), dpi=130)
    plt.close(fig)
    return resume


# --------------------------------------------------------------------------- #
#  FIG 7 -- cout & accessibilite : flotte vs GPU cloud vs serveur
# --------------------------------------------------------------------------- #
def fig7_cout(seq, n_volontaires=30):
    A = ASSUMPTIONS
    steps_to_target = next((r["grad_steps"] for r in seq["history"]
                            if r["accuracy"] >= seq["target_accuracy"]), seq["total_grad_steps"])
    tg = seq["per_gradient_seconds"]
    heures_demo = steps_to_target * tg / 3600.0   # cout reel de CETTE demo (tres petit)
    # le cout absolu d'une demo de 20s est negligeable ; on exprime donc le modele
    # sur un entrainement REALISTE (heures-PC, editable). Le RATIO entre options est
    # le message, et il est invariant d'echelle.
    heures_pc = A["entrainement_realiste_heures_pc"]

    # flotte volontaire : electricite marginale seulement, capital ~ 0
    p_moy_w = (A["mix_flotte_smartphone"] * A["puissance_smartphone_w"]
               + (1 - A["mix_flotte_smartphone"]) * A["puissance_pc_w"])
    heures_mur = heures_pc / n_volontaires
    e_flotte = n_volontaires * p_moy_w * A["fraction_marginale_volontaire"] * heures_mur / 1000.0
    c_flotte = e_flotte * A["prix_electricite_usd_kwh"]

    # GPU cloud loue
    h_gpu = heures_pc / A["acceleration_gpu_vs_pc"]
    c_gpu = h_gpu * A["prix_gpu_cloud_usd_h"]
    e_gpu = A["puissance_gpu_cloud_w"] * h_gpu / 1000.0

    # serveur dedie
    h_srv = heures_pc / A["acceleration_serveur_vs_pc"]
    c_srv = h_srv * A["cout_amorti_serveur_usd_h"]
    e_srv = A["puissance_serveur_w"] * h_srv / 1000.0

    labels = [f"Flotte volontaire\n({n_volontaires} appareils)", "GPU cloud loue", "Serveur dedie"]
    couts = [c_flotte, c_gpu, c_srv]; energies = [e_flotte, e_gpu, e_srv]
    cols = [C_FLEET, C_CLOUD, C_SERVER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    b = axes[0].bar(labels, couts, color=cols)
    axes[0].bar_label(b, fmt="$%.3f"); axes[0].set_ylabel("Cout pour atteindre la cible (USD)")
    axes[0].set_title("Cout monetaire")
    b = axes[1].bar(labels, energies, color=cols)
    axes[1].bar_label(b, fmt="%.3f kWh"); axes[1].set_ylabel("Energie (kWh)")
    axes[1].set_title("Energie consommee")
    fig.suptitle(f"Cout & accessibilite pour un entrainement realiste "
                 f"(~{heures_pc:.0f} heures-PC, editable)\n"
                 "la flotte ne paie que l'electricite marginale d'appareils deja possedes "
                 "(capital ~ 0)", fontsize=11)
    hyp = (f"Hypotheses : elec {A['prix_electricite_usd_kwh']} $/kWh | "
           f"GPU {A['prix_gpu_cloud_usd_h']} $/h | serveur {A['cout_amorti_serveur_usd_h']} $/h | "
           f"smartphone {A['puissance_smartphone_w']} W | PC {A['puissance_pc_w']} W")
    fig.text(0.5, -0.02, hyp, ha="center", fontsize=8, color="#555")
    fig.savefig(os.path.join(RESULTS, "fig7_cout_accessibilite.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {"heures_pc_demo": heures_demo, "heures_pc_modele": heures_pc,
            "cout_usd": {"flotte": c_flotte, "gpu_cloud": c_gpu, "serveur": c_srv},
            "energie_kwh": {"flotte": e_flotte, "gpu_cloud": e_gpu, "serveur": e_srv}}


def main():
    cfg = DEFAULT
    job = AttentionCNN3DJob()

    print("Chargement des executions...")
    seq = ensure_sequential()
    dist = _load("distributed_run.json")
    if not dist:
        print("  (distributed_run.json absent -> figures heterogeneite ignorees ;")
        print("   lancer 'python scripts/run_simulation.py' pour les obtenir)")

    M = {"hypotheses_cout": ASSUMPTIONS}
    print("Fig 1 : qualite medicale...");      M["qualite"] = fig1_qualite(seq)
    print("Fig 2 : plus-value vs sequentiel..."); M["plus_value"] = fig2_plus_value(seq)
    print("Fig 3 : passage a l'echelle...");    M["echelle"] = fig3_echelle(seq)
    print("Fig 4 : bande passante...");         M["bande_passante"] = fig4_bande_passante(job)
    print("Fig 5 : heterogeneite...");          M["heterogeneite"] = fig5_heterogeneite(dist)
    print("Fig 6 : tolerance aux pannes (~1 min)..."); M["tolerance"] = fig6_tolerance(job, cfg)
    print("Fig 7 : cout & accessibilite...");   M["cout"] = fig7_cout(seq)

    with open(os.path.join(RESULTS, "metrics_memoire.json"), "w") as f:
        json.dump(M, f, indent=2)

    print("\n=== RESUME POUR LE MEMOIRE ===")
    print(f"Qualite : precision {M['qualite']['precision_finale']:.1%}, "
          f"top-3 {M['qualite']['top3_finale']:.1%}, F1 {M['qualite']['f1_finale']:.2f}")
    print(f"Phase : {M['qualite']['tours_initial']:.0f} -> "
          f"{M['qualite']['tours_final']:.1f} étapes de travail")
    if M["plus_value"]["acceleration_K_max"]:
        print(f"Plus-value : ~x{M['plus_value']['acceleration_K_max']:.0f} plus rapide "
              f"que le sequentiel (10 volontaires, modelise)")
    print(f"Transport : x{M['bande_passante']['reduction_top10']:.0f} de reduction "
          f"(gradients fp16+top10% vs poids fp32)")
    print(f"Cout cible : flotte ${M['cout']['cout_usd']['flotte']:.3f} | "
          f"GPU ${M['cout']['cout_usd']['gpu_cloud']:.3f} | "
          f"serveur ${M['cout']['cout_usd']['serveur']:.3f}")
    print("\nFigures + metrics_memoire.json dans results/")


if __name__ == "__main__":
    main()
