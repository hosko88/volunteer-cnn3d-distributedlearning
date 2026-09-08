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

from jobs.progressive.models_pytorch import (
    CNN2D, CNN3D, CNN3DAttention, params_to_vector, vector_to_params,
)
from jobs.progressive.data_providers import CIFAR10Provider, ModelNet40Provider
from framework.compression import encode_vector, decode_vector, raw_size_bytes, compression_ratio

app = Flask(__name__)
lock = threading.Lock()

PHASES = {
    1: {"job": "Phase1-CIFAR10-CNN2D", "n_classes": 10, "default_batch": 64},
    2: {"job": "Phase2-ModelNet40-CNN3D", "n_classes": 40, "default_batch": 16},
    3: {"job": "Phase3-ModelNet40-CNN3D-Attention", "n_classes": 40, "default_batch": 16},
}

# Reference de temps sequentiel mesuree reellement (sanity_check_cnn2d_local.py,
# CPU, sans distribution) : 1 epoque complete sur les 50 000 images CIFAR-10.
# Aucune reference centralisee equivalente n'a ete mesuree pour ModelNet40 (phase 2/3)
# a ce stade -- le speedup n'est donc calcule que pour la phase 1, honnetement.


def build_model(phase: int, n_classes: int):
    if phase == 1:
        return CNN2D(n_classes=n_classes)
    if phase == 2:
        return CNN3D(n_classes=n_classes, in_channels=1)
    if phase == 3:
        return CNN3DAttention(n_classes=n_classes, in_channels=1)
    raise ValueError(f"Phase inconnue : {phase} (attendu : 1, 2 ou 3)")


def build_provider(phase: int):
    if phase == 1:
        return CIFAR10Provider()
    if phase in (2, 3):
        return ModelNet40Provider()
    raise ValueError(f"Phase inconnue : {phase} (attendu : 1, 2 ou 3)")


def stable_shard_id(client_id: str, n_shards: int) -> int:
    h = hashlib.sha256(client_id.encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, n_shards)


class FedAvgState:
    # Reference mesuree sur ce projet (sanity_check_cnn2d_local.py) :
    # 1 epoque complete (50000 images, CNN2D, CPU) = 274.4s => 0.005488 s/image.
    # Sert de base "sequentielle" pour calculer le speedup en phase 1.
    REF_SECONDS_PER_IMAGE = {1: 274.4 / 50000}

    def __init__(self, phase: int, n_shards: int, max_rounds: int, local_epochs: int,
                 batch_size: int, lr: float, target_accuracy: float,
                 round_timeout: float, ref_seconds_per_image: float | None = None):
        info = PHASES[phase]
        self.phase = phase
        self.job = info["job"]
        self.n_classes = info["n_classes"]
        self.n_shards = n_shards
        self.max_rounds = max_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.target_accuracy = target_accuracy
        self.round_timeout = round_timeout
        # Reference explicite (--ref-seconds-per-image) sinon celle connue pour la
        # phase, sinon None (speedup non calculable, affiche "--" plutot qu'invente)
        self.ref_seconds_per_image = ref_seconds_per_image or self.REF_SECONDS_PER_IMAGE.get(phase)

        self.data = build_provider(phase)
        self.model = build_model(phase, self.n_classes)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.theta = params_to_vector(self.model)

        self.round_id = 0
        self.round_start = time.time()
        self.first_submission_time = None   # le compte a rebours du timeout ne
                                             # demarre qu'a la 1ere soumission du
                                             # round, pas au demarrage serveur
        self.submissions = {}          # client_id -> (delta_vec, n_samples)
        self.assigned_shards = {}      # client_id -> shard_id
        self.finished = False
        self.history = []
        self.start_time = time.time()

        self.bytes_uploaded_total = 0
        self.bytes_downloaded_total = 0
        self.samples_processed_total = 0
        self.client_stats = {}   # client_id -> {rounds, last_local_time, last_n_samples,
                                  #               shard_id, total_local_time, total_samples}

        self.stale_submissions = 0     # soumissions arrivees apres agregation du round -> gaspillees
        self.cum_sequential_seconds = 0.0  # temps "sequentiel equivalent" cumule
        self.cum_wall_seconds = 0.0        # temps reel cumule (calcul utile uniquement, hors attente)

    def assign_shard(self, client_id: str) -> int:
        if client_id not in self.assigned_shards:
            self.assigned_shards[client_id] = stable_shard_id(client_id, self.n_shards)
        return self.assigned_shards[client_id]

    def maybe_aggregate(self):
        """Agrege si tous les shards ont soumis, ou si le timeout du round est depasse
        (decompte a partir de la 1ere soumission recue) avec au moins une soumission.
        Doit etre appele avec le lock deja pris."""
        if self.finished:
            return
        n_sub = len(self.submissions)
        if n_sub == 0:
            return
        timed_out = (self.first_submission_time is not None
                     and (time.time() - self.first_submission_time) > self.round_timeout)
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
        x, y = self.data.sample_eval()
        with torch.no_grad():
            xt = torch.from_numpy(x)
            pred = self.model(xt).argmax(dim=1).numpy()
        acc = float((pred == y).mean())

        # --- Duree reelle de ce round : depuis la 1ere soumission (comme le timeout),
        #     pas depuis le demarrage du round, pour ne pas compter le temps d'attente
        #     avant qu'un volontaire ne se manifeste. ---
        round_wall_seconds = (time.time() - self.first_submission_time) if self.first_submission_time else 0.0

        # --- Speedup / efficacite : comparaison au temps qu'aurait pris le meme
        #     volume de travail en sequentiel, sur UNE seule machine. ---
        images_processed = total_samples * self.local_epochs
        if self.ref_seconds_per_image and round_wall_seconds > 0:
            sequential_seconds = images_processed * self.ref_seconds_per_image
            speedup = sequential_seconds / round_wall_seconds
            efficiency = speedup / max(1, n_sub)
            self.cum_sequential_seconds += sequential_seconds
            self.cum_wall_seconds += round_wall_seconds
        else:
            sequential_seconds = None
            speedup = None
            efficiency = None

        elapsed = time.time() - self.start_time
        speedup_str = f"x{speedup:.2f}" if speedup is not None else "--"
        print(f"[FedAvg] round {self.round_id} termine "
              f"({n_sub}/{self.n_shards} volontaires, {'TIMEOUT' if timed_out else 'complet'}) "
              f"-> precision={acc*100:.2f}%  duree_round={round_wall_seconds:.0f}s  "
              f"speedup={speedup_str}  elapsed_total={elapsed:.0f}s")

        self.history.append({
            "round": self.round_id,
            "accuracy": acc,
            "participants": n_sub,
            "elapsed": elapsed,
            "complete": not timed_out,
            "round_duration_s": round_wall_seconds,
            "images_processed": images_processed,
            "speedup": speedup,
            "efficiency": efficiency,
        })

        self.round_id += 1
        self.submissions = {}
        self.first_submission_time = None
        self.round_start = time.time()

        if self.round_id >= self.max_rounds or acc >= self.target_accuracy:
            self.finished = True
            print(f"[FedAvg] TERMINE en {elapsed:.0f}s ({elapsed/60:.1f} min) "
                  f"-- precision finale={acc*100:.2f}%")


state: FedAvgState | None = None


@app.route("/fedavg/config")
def fedavg_config():
    return jsonify({
        "phase": state.phase,
        "job": state.job,
        "n_classes": state.n_classes,
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
    local_time = float(d.get("local_time", 0.0))
    delta = decode_vector(d["delta"])
    nbytes = len(d["delta"])  # approx (base64), suffisant pour les metriques

    with lock:
        if state.finished:
            return jsonify({"status": "finished"})
        if round_id != state.round_id:
            # Soumission perimee (le round a deja avance) -- on l'ignore proprement,
            # mais on la compte : c'est du calcul reellement gaspille.
            state.stale_submissions += 1
            print(f"[FedAvg] soumission perimee de {client_id} pour le round {round_id} "
                  f"(le serveur en est au round {state.round_id}) -- ignoree")
            return jsonify({"status": "stale", "current_round": state.round_id})
        if state.first_submission_time is None:
            state.first_submission_time = time.time()
        state.submissions[client_id] = (delta, n_samples)

        cs = state.client_stats.setdefault(client_id, {
            "rounds": 0, "last_local_time": None, "last_n_samples": None,
            "shard_id": state.assigned_shards.get(client_id),
            "total_local_time": 0.0, "total_samples": 0,
        })
        cs["rounds"] += 1
        cs["last_local_time"] = local_time
        cs["last_n_samples"] = n_samples
        cs["total_local_time"] += local_time
        cs["total_samples"] += n_samples

        state.bytes_uploaded_total += nbytes
        print(f"[FedAvg] reçu round {round_id} de {client_id} "
              f"({len(state.submissions)}/{state.n_shards} pour ce round, "
              f"local_time={local_time:.0f}s pour {n_samples} images)")
        state.maybe_aggregate()
        return jsonify({"status": "ok"})


@app.route("/fedavg/status")
def fedavg_status():
    with lock:
        raw_equiv = raw_size_bytes(state.n_params) * 2 * max(1, state.round_id) * state.n_shards
        comp_ratio = (raw_equiv / state.bytes_uploaded_total) if state.bytes_uploaded_total else 0.0
        cum_speedup = (state.cum_sequential_seconds / state.cum_wall_seconds
                       if state.cum_wall_seconds > 0 else None)
        cum_efficiency = (cum_speedup / state.n_shards) if cum_speedup is not None else None
        return jsonify({
            "phase": state.phase,
            "job": state.job,
            "n_params": state.n_params,
            "round": state.round_id,
            "max_rounds": state.max_rounds,
            "finished": state.finished,
            "history": state.history,
            "n_shards": state.n_shards,
            "submitted_this_round": len(state.submissions),
            "target_accuracy": state.target_accuracy,
            "elapsed": time.time() - state.start_time,
            "bytes_uploaded_total": state.bytes_uploaded_total,
            "bytes_downloaded_total": state.bytes_downloaded_total,
            "raw_equivalent_bytes": raw_equiv,
            "compression_ratio": comp_ratio,
            "stale_submissions": state.stale_submissions,
            "ref_seconds_per_image": state.ref_seconds_per_image,
            "cum_speedup": cum_speedup,
            "cum_efficiency": cum_efficiency,
            "client_stats": state.client_stats,
        })


@app.route("/fedavg/export.json")
def fedavg_export_json():
    with lock:
        cum_speedup = (state.cum_sequential_seconds / state.cum_wall_seconds
                       if state.cum_wall_seconds > 0 else None)
        return jsonify({
            "phase": state.phase,
            "job": state.job,
            "n_params": state.n_params,
            "n_shards": state.n_shards,
            "local_epochs": state.local_epochs,
            "batch_size": state.batch_size,
            "max_rounds": state.max_rounds,
            "target_accuracy": state.target_accuracy,
            "finished": state.finished,
            "elapsed_total": time.time() - state.start_time,
            "bytes_uploaded_total": state.bytes_uploaded_total,
            "bytes_downloaded_total": state.bytes_downloaded_total,
            "stale_submissions": state.stale_submissions,
            "ref_seconds_per_image": state.ref_seconds_per_image,
            "cum_speedup": cum_speedup,
            "cum_efficiency": (cum_speedup / state.n_shards) if cum_speedup is not None else None,
            "client_stats": state.client_stats,
            "history": state.history,
        })


@app.route("/fedavg/export.csv")
def fedavg_export_csv():
    from flask import Response
    lines = ["round,accuracy,participants,n_shards,complete,round_duration_s,"
             "images_processed,speedup,efficiency,elapsed_total_s"]
    with lock:
        for h in state.history:
            sp = f"{h['speedup']:.3f}" if h.get('speedup') is not None else ""
            ef = f"{h['efficiency']:.3f}" if h.get('efficiency') is not None else ""
            lines.append(f"{h['round']},{h['accuracy']:.6f},{h['participants']},"
                          f"{state.n_shards},{h.get('complete', True)},"
                          f"{h.get('round_duration_s', 0):.1f},{h.get('images_processed', 0)},"
                          f"{sp},{ef},{h['elapsed']:.1f}")
    return Response("\n".join(lines), mimetype="text/csv",
                     headers={"Content-Disposition": "attachment; filename=fedavg_metrics.csv"})


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Tableau de bord FedAvg</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body { background:#0b0f14; color:#e6edf3; font-family: ui-monospace, monospace; margin:0; padding:24px; }
  h1 { font-size:20px; letter-spacing:1px; }
  .sub { color:#8b949e; margin-bottom:16px; }
  .badge { display:inline-block; padding:4px 12px; border-radius:16px; border:1px solid #2ea043;
           color:#2ea043; font-size:13px; margin-right:8px; }
  .badge.wait { border-color:#d29922; color:#d29922; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:20px 0; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:14px 16px; }
  .card .label { color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
  .card .value { font-size:26px; margin-top:6px; }
  .btns a { color:#58a6ff; text-decoration:none; border:1px solid #30363d; padding:6px 14px;
            border-radius:6px; margin-right:8px; font-size:13px; }
  canvas { background:#161b22; border-radius:10px; padding:10px; margin-top:10px; }
  table { width:100%; border-collapse:collapse; margin-top:20px; background:#161b22; border-radius:10px; overflow:hidden; }
  th, td { text-align:left; padding:10px 14px; border-bottom:1px solid #30363d; font-size:13px; }
  th { color:#8b949e; text-transform:uppercase; font-size:11px; letter-spacing:1px; }
</style>
</head>
<body>
<h1 id="job">Chargement...</h1>
<div class="sub" id="sub"></div>
<div id="badges"></div>
<div class="btns" style="margin-top:12px;">
  <a href="/fedavg/export.json">Exporter JSON</a>
  <a href="/fedavg/export.csv">Exporter CSV</a>
</div>
<div class="grid" id="cards"></div>
<canvas id="chart" height="90"></canvas>
<table><tbody id="vol-table"></tbody></table>

<script>
let chart = null;

function card(label, value) {
  return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

async function refresh() {
  const r = await fetch('/fedavg/status');
  const s = await r.json();

  document.getElementById('job').textContent = s.job + '  (' + s.n_params.toLocaleString() + ' parametres)';
  document.getElementById('sub').textContent =
    'Round ' + s.round + ' / ' + s.max_rounds + '  --  ' + s.n_shards + ' shard(s)';

  const badges = s.finished
    ? '<span class="badge">TERMINE</span>'
    : '<span class="badge wait">en cours -- ' + s.submitted_this_round + '/' + s.n_shards + ' pour ce round</span>';
  document.getElementById('badges').innerHTML = badges;

  const lastAcc = s.history.length ? (s.history[s.history.length-1].accuracy*100).toFixed(2) : '--';
  const elapsedMin = (s.elapsed/60).toFixed(1);
  const ratio = s.compression_ratio ? s.compression_ratio.toFixed(1) : '--';
  const upMB = (s.bytes_uploaded_total/1e6).toFixed(1);
  const downMB = (s.bytes_downloaded_total/1e6).toFixed(1);
  const speedup = s.cum_speedup !== null ? ('&times;' + s.cum_speedup.toFixed(2)) : 'N/A';
  const eff = s.cum_efficiency !== null ? (s.cum_efficiency*100).toFixed(0) + '%' : 'N/A';

  document.getElementById('cards').innerHTML =
      card('Precision actuelle', lastAcc + '%')
    + card('Cible', (s.target_accuracy*100).toFixed(0) + '%')
    + card('Temps ecoule', elapsedMin + ' min')
    + card('Rounds', s.round + ' / ' + s.max_rounds)
    + card('Speedup cumule', speedup)
    + card('Efficacite', eff)
    + card('Envoye (deltas)', upMB + ' Mo')
    + card('Recu (poids)', downMB + ' Mo')
    + card('Facteur de reduction', '&times;' + ratio)
    + card('Soumissions perimees', s.stale_submissions);

  const rows = Object.entries(s.client_stats || {}).map(([cid, cs]) => {
    const avgSpeed = cs.total_local_time > 0 ? (cs.total_samples/cs.total_local_time).toFixed(0) : '--';
    return `<tr><td>${cid}</td><td>shard ${cs.shard_id}</td><td>${cs.rounds}</td>`
         + `<td>${cs.last_local_time !== null ? cs.last_local_time.toFixed(0)+'s' : '--'}</td>`
         + `<td>${avgSpeed} img/s</td></tr>`;
  }).join('');
  document.getElementById('vol-table').innerHTML =
    '<tr><th>Volontaire</th><th>Shard</th><th>Rounds soumis</th><th>Dernier temps local</th><th>Debit moyen</th></tr>' + rows;

  const labels = s.history.map(h => 'R' + h.round);
  const accData = s.history.map(h => (h.accuracy*100).toFixed(2));
  const partData = s.history.map(h => h.participants);

  if (!chart) {
    const ctx = document.getElementById('chart').getContext('2d');
    chart = new Chart(ctx, {
      type: 'line',
      data: { labels: labels, datasets: [
        { label: 'Precision (%)', data: accData, borderColor:'#2ea043', yAxisID:'y', tension:0.2 },
        { label: 'Volontaires participants', data: partData, borderColor:'#58a6ff', yAxisID:'y1', tension:0.2 },
      ]},
      options: {
        scales: {
          y: { position:'left', ticks:{color:'#8b949e'}, grid:{color:'#30363d'} },
          y1:{ position:'right', ticks:{color:'#8b949e'}, grid:{drawOnChartArea:false} },
          x: { ticks:{color:'#8b949e'}, grid:{color:'#30363d'} },
        },
        plugins: { legend:{ labels:{ color:'#e6edf3' } } },
      }
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = accData;
    chart.data.datasets[1].data = partData;
    chart.update();
  }

  if (!s.finished) setTimeout(refresh, 3000);
}
refresh();
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return DASHBOARD_HTML


def main():
    global state
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                     help="1=CIFAR-10 CNN2D, 2=ModelNet40 CNN3D, 3=ModelNet40 CNN3D+Attention")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=None,
                     help="Par defaut : 64 en phase 1, 16 en phase 2/3.")
    ap.add_argument("--shards", type=int, default=3, help="Nombre de volontaires attendus (partitions du dataset).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-accuracy", type=float, default=0.75)
    ap.add_argument("--round-timeout", type=float, default=600.0,
                     help="Secondes avant d'agreger meme si tous les volontaires n'ont pas soumis "
                          "(decompte a partir de la 1ere soumission du round).")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    batch_size = args.batch_size or PHASES[args.phase]["default_batch"]

    state = FedAvgState(
        phase=args.phase, n_shards=args.shards, max_rounds=args.rounds,
        local_epochs=args.local_epochs, batch_size=batch_size, lr=args.lr,
        target_accuracy=args.target_accuracy, round_timeout=args.round_timeout,
    )

    print("================================================================")
    print("  SERVEUR FEDAVG -- rounds d'entrainement local")
    print("================================================================")
    print(f"  Phase         : {args.phase} -- {state.job}")
    print(f"  Modele        : {state.n_params:,} parametres, {state.n_classes} classes")
    print(f"  Rounds prevus : {args.rounds}")
    print(f"  Shards        : {args.shards} (= nombre de volontaires attendus)")
    print(f"  Epoques locales/round : {args.local_epochs}  |  batch={batch_size}")
    print(f"  Port          : {args.port}")
    print("================================================================")
    print(f"  Tableau de bord     : http://<IP>:{args.port}/")
    print(f"  Commande volontaire : python scripts/run_volunteer_fedavg.py "
          f"--server http://<IP>:{args.port} --device <nom>")
    print("================================================================")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
