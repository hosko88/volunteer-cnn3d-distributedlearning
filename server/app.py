"""
VolunteerRL Framework
----------------------------------------------

Using Volunteer Computing for Distributed Learning
with Convolutional Neural Networks:
A Progressive Approach Toward 3D Imaging

Developed by:
Ngono Ndjana Clement

Departement d'informatique (University of Yaoundé I)

2026
"""

"""
Serveur HTTP de coordination (Flask)
====================================
Expose l'API REST utilisee par les volontaires et sert le tableau de bord
temps reel. Le transport (poids descendants, gradients montants) passe par des
chaines base64 compressees.
"""

import os
import io
import base64
import hashlib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from framework.coordinator import Coordinator

HERE = os.path.dirname(os.path.abspath(__file__))

# Le serveur ne démarre pas immédiatement l'entraînement.
# Les volontaires attendront que cette variable passe à True.
TRAINING_STARTED = False


def stable_shard_id(client_id: str, n_shards: int) -> int:
    """Attribution stable et deterministe d'une partition a un volontaire
    (le meme client_id recoit toujours la meme partition)."""
    h = hashlib.md5(client_id.encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, n_shards)


def _encode_shard(x: np.ndarray, y: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, x=x.astype(np.float16), y=y.astype(np.int64))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(job_heavy, cfg, job_light=None, auto_start=False):
    """
    job_heavy : job principal (ex. Phase1/2/3 PyTorch), servi aux clients
                capables (info["has_torch"] == True côté volontaire).
    job_light : job leger optionnel (ex. AttentionCNN3DJob NumPy), servi aux
                clients sans PyTorch (ex. smartphone Termux). Si None, tous
                les clients recoivent job_heavy (comportement d'avant, non
                heterogene) -- retrocompatible avec les scripts existants.

    Systeme heterogene : deux entrainements distribues independants tournent
    en parallele (deux ParameterServer/Coordinator distincts -- impossible
    de fusionner les gradients de deux architectures differentes dans un
    meme vecteur de parametres). Un client est affecte a l'une ou l'autre
    piste selon sa capacite declaree (has_torch), de facon stable d'un appel
    a l'autre (memorisee dans client_track).
    """
    global TRAINING_STARTED
    TRAINING_STARTED = bool(auto_start)

    app = Flask(__name__)
    coord_heavy = Coordinator(job_heavy, cfg)
    coord_light = Coordinator(job_light, cfg) if job_light is not None else None

    # Retrocompatibilite : app.coord pointe vers la piste principale (utilise
    # par les scripts/tests existants qui ne connaissent qu'une seule piste).
    app.coord = coord_heavy
    app.coord_heavy = coord_heavy
    app.coord_light = coord_light

    # Piste choisie par client (memoire simple, en RAM) : {client_id: "heavy"|"light"}
    client_track = {}

    def pick_track(info: dict) -> str:
        if coord_light is None:
            return "heavy"
        return "heavy" if info.get("has_torch") else "light"

    def coord_for(track: str) -> Coordinator:
        return coord_heavy if track == "heavy" else coord_light

    def track_done(coord) -> bool:
        """Une piste est consideree terminee si elle a fini son travail, OU
        si elle n'a jamais recu le moindre volontaire (--no-light-track
        desactive, ou pool 100% heavy / 100% light) : sinon `finished`
        global resterait bloque a False indefiniment (total==0 pour
        toujours dans _maybe_finalize)."""
        return coord is None or coord.finished or not coord.has_had_client

    def all_tracks_finished() -> bool:
        # Il faut qu'AU MOINS une piste ait reellement termine (sinon on
        # dirait "termine" avant meme d'avoir commence, les deux pistes
        # etant "sans volontaire" au tout debut), ET qu'aucune piste
        # encore active/peuplee ne soit en cours.
        really_finished = coord_heavy.finished or (coord_light is not None and coord_light.finished)
        return really_finished and track_done(coord_heavy) and track_done(coord_light)

    @app.get("/")
    def dashboard():
        return send_from_directory(HERE, "dashboard.html")

    @app.get("/kb")
    def kb():
        # specification du job + reglages de transport (le volontaire reconstruit le job)
        has_torch = request.args.get("has_torch", "1") == "1"
        track = pick_track({"has_torch": has_torch})
        job = job_heavy if track == "heavy" else job_light
        return jsonify({
            "kb": job.kb_spec(),
            "transport": {"dtype": cfg.transport.dtype, "topk": cfg.transport.topk},
            "job_type": job.__class__.__name__,
            "track": track,
            "n_shards": cfg.train.n_shards,
        })

    @app.get("/shard")
    def shard():
        """
        DEPRECATED.
        Les partitions (shards) ne sont plus utilisées.
        Le serveur envoie uniquement les mini-batches nécessaires
        dans chaque tâche (batch_payload). Le volontaire ne télécharge
        jamais le dataset ni une grosse partition.
        """
        return jsonify({
            "error": "DEPRECATED: Les partitions (shards) ne sont plus utilisées. "
                     "Le serveur envoie uniquement les mini-batches nécessaires "
                     "dans chaque tâche (champ batch_payload)."
        }), 410

    @app.post("/request_work")
    def request_work():
        body = request.get_json(force=True)

        client_id = body["client_id"]
        info = body.get("info", {})
        power = body.get("power", 1)

        track = client_track.get(client_id) or pick_track(info)
        client_track[client_id] = track

        print(
            f"[POST /request_work] client={client_id} track={track} "
            f"device={info.get('device')} os={info.get('os')} "
            f"cpu={info.get('cpu')} ram={info.get('ram_gb')}Go power={power}"
        )

        out = coord_for(track).request_work(client_id, info, power)

        print(
            f"[SEND] client={client_id} track={track} "
            f"tasks={len(out.get('tasks', []))} "
            f"finished={out.get('finished')}"
        )

        return jsonify(out)

    @app.post("/report")
    def report():
        body = request.get_json(force=True)

        client_id = body["client_id"]
        results = body.get("results", [])
        track = client_track.get(client_id, "heavy")

        print(f"[POST /report] client={client_id} track={track} results={len(results)}")

        for r in results:
            lm = r.get("local_metrics", {})
            print(
                f"  -> task={r.get('task_id')} "
                f"duration={lm.get('duration_seconds', 0):.2f}s "
                f"samples={r.get('n_samples')}"
            )

        coord_for(track).report_gradients(client_id, results)

        return jsonify({"ok": True, "finished": all_tracks_finished()})

    @app.post("/client_comm_metrics")
    def client_comm_metrics():
        body = request.get_json(force=True)

        client_id = body.get("client_id")
        task_ids = body.get("task_ids", [])
        report_seconds = body.get("report_seconds", 0.0)
        tasks_count = body.get("tasks_count", len(task_ids))
        track = client_track.get(client_id, "heavy")

        coord_for(track).record_client_comm_metrics(
            client_id, task_ids, report_seconds, tasks_count
        )

        return jsonify({"ok": True})

    @app.get("/status")
    def status():
        s = coord_heavy.status()
        if coord_light is not None:
            s["light_track"] = coord_light.status()
        s["heterogeneous"] = coord_light is not None
        return jsonify(s)

    @app.post("/phase")
    def phase_selector():
        # Le changement de phase ne concerne que la piste lourde (PyTorch) :
        # la piste legere (NumPy) reste toujours sur son unique job (Phase 1).
        body = request.get_json(force=True)
        phase_name = body.get("phase")
        if coord_heavy.set_phase(phase_name):
            return jsonify({"ok": True, "phase": coord_heavy.current_phase})
        return jsonify({"ok": False, "phase": coord_heavy.current_phase})

    @app.get("/export")
    def export_session():
        out = coord_heavy.export_snapshot()
        if coord_light is not None:
            out["light_track"] = coord_light.export_snapshot()
        return jsonify(out)

    @app.post("/start_training")
    def start_training():
        global TRAINING_STARTED
        TRAINING_STARTED = True

        # Le chronomètre démarre exactement quand tu cliques sur DÉMARRER
        coord_heavy.start_timer()
        if coord_light is not None:
            coord_light.start_timer()

        return jsonify({"started": True})

    @app.get("/training_status")
    def training_status():
        return jsonify({"started": TRAINING_STARTED})

    return app
