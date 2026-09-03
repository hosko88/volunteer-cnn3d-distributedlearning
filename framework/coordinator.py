"""
VolunteerRL Framework
----------------------------------------------

Volunteer Computing for Distributed Learning
with Convolutional Neural Networks
for Progressive 3D Imaging

Developed by:
Ngono Ndjana Clement

Departement d'informatique (University of Yaoundé I)

2026
"""
"""
Coordinateur
============
Chef d'orchestre du calcul distribue. A chaque epoque :
  1. genere les sous-taches et les confie a l'ordonnanceur ;
  2. distribue le travail aux volontaires (avec les parametres globaux) ;
  3. assimile les gradients reçus dans le serveur de parametres (Adam) ;
  4. des que `completion_fraction` des sous-taches sont rendues, FINALISE l'epoque
     sans attendre les retardataires (asynchronisme + tolerance aux pannes),
     evalue le modele global, enregistre les metriques, et decide de continuer
     ou de s'ARRETER (convergence atteinte -> le serveur signale la fin).
"""

import math
import time
import threading

from .parameter_server import ParameterServer
from .scheduler import Scheduler
from .work_generator import WorkGenerator


class Coordinator:
    def __init__(self, job, cfg):
        self.job = job
        self.cfg = cfg
        self.ps = ParameterServer(job, cfg.transport)
        self.sched = Scheduler(cfg)
        self.wg = WorkGenerator(job, cfg)
        self.epoch = 0
        self.has_had_client = False  # cf. request_work() : evite un blocage
                                      # de `finished` global si cette piste
                                      # (heavy ou light) n'a jamais recu
                                      # aucun volontaire (voir server/app.py)
        self.epsilon = cfg.model.epsilon_start
        self.history = []
        self.phase_history = {
            "Phase 1 - CIFAR-10": [],
            "Phase 2 - ModelNet40": [],
            "Phase 3 - ShapeNet": [],
        }
        self.current_phase = "Phase 1 - CIFAR-10"
        self.run_id = f"run-{int(time.time() * 1000)}"
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.epoch_start = None
        self._finalizing = False
        self.lock = threading.Lock()
        # metriques locales agregees de l'epoque courante
        self._epoch_local_acc = []
        self._epoch_local_turns = []
        self.events = []
    
    def start_timer(self):
        """
        Démarre le chronomètre et lance la première époque
        au moment où l'utilisateur clique sur DÉMARRER.
        """
        if self.start_time is None:
            self.start_time = time.time()
            self.end_time = None
            self._begin_epoch(1)

    def add_event(self, event_type, message):
        """
        Ajoute un événement visible dans le journal temps réel du dashboard.
        """
        self.events.append({
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "message": message
        })

        if len(self.events) > 200:
            self.events = self.events[-200:]

    # ----- demarrage ----- #
    def start(self):
        self.start_time = None
        self.end_time = None
        self.finished = False
        self.run_id = f"run-{int(time.time() * 1000)}"

    def set_phase(self, phase_name):
        if phase_name in self.phase_history:
            self.current_phase = phase_name
            return True
        return False

    def export_snapshot(self):
        status = self.status()
        return {
            "run_id": status["run_id"],
            "phase": status["current_phase"],
            "phase_history": status["phase_history"],
            "history": status["history"],
            "bandwidth": status["bandwidth"],
            "scheduler": status["scheduler"],
            "distributed_metrics": status["distributed_metrics"],
        }
   

    def _begin_epoch(self, e):
        self.epoch = e
        self.epoch_start = time.time()
        self._finalizing = False
        self._epoch_local_acc = []
        self._epoch_local_turns = []
        tasks = self.wg.create_epoch(e, self.epsilon)
        self.sched.load_epoch(tasks)

    # ----- API appelee par les volontaires ----- #
    def request_work(self, client_id, info, power):
        self.has_had_client = True
        self.sched.register(client_id, info, power)
        if self.finished:
            return {"tasks": [], "finished": True}
        batch = self.sched.assign(client_id)

        # Enrichissement côté serveur : chaque tâche reçoit son mini-batch.
        # Le volontaire ne télécharge JAMAIS le dataset ni une partition.
        #
        # IMPORTANT : si l'enrichissement échoue, on NE DOIT JAMAIS envoyer la
        # tâche "nue" au volontaire. Le chemin de repli de compute_gradient()
        # (self.data.sample_batch(...)) déclencherait alors, côté volontaire,
        # un téléchargement/une préparation COMPLÈTE du dataset (CIFAR-10,
        # ModelNet40) -- exactement ce que ce système doit interdire. On
        # remet donc la tâche en file d'attente pour une nouvelle tentative,
        # plutôt que de dégrader silencieusement le protocole.
        enriched = []
        for task in batch:
            try:
                t = self.job.enrich_task_with_data(task)
                if "batch_payload" not in t:
                    raise RuntimeError(
                        "enrich_task_with_data n'a pas produit de batch_payload"
                    )
                enriched.append(t)
            except Exception as e:
                with self.sched.lock:
                    self.sched.inflight.pop(task["task_id"], None)
                    self.sched.pending.appendleft(task["task_id"])
                self.add_event(
                    "ERROR",
                    f"enrich_task a échoué pour {task.get('task_id')} : {e} "
                    f"-- tâche remise en file (jamais envoyée sans données)",
                )

        if enriched:
            device = info.get("device", client_id)
            for task in enriched:
                self.add_event(
                    "SEND_TASK",
                    f"{device} reçoit {task['task_id']}"
                )
        else:
            self.add_event(
                "NO_TASK",
                f"{info.get('device', client_id)} attend : aucune sous-tâche disponible"
            )
        payload, version = self.ps.params_payload()
        return {"tasks": enriched, "params": payload, "params_version": version,
                "finished": False}

    def report_gradients(self, client_id, results):
        for r in results:
            lm = r.get("local_metrics") or {}

            # IMPORTANT : le scheduler sait déjà déduire une tâche déjà
            # terminée (completed_ids) -- notamment après une réattribution
            # pour cause d'expiration, quand le volontaire d'origine répond
            # quand même en retard. Il FAUT consulter ce résultat AVANT
            # d'appliquer le gradient : sinon la même sous-tâche est comptée
            # deux fois dans theta (biais de la mise à jour), même si les
            # statistiques d'avancement, elles, restaient correctes.
            accepted = self.sched.report(
                client_id,
                r["task_id"],
                lm.get("duration_seconds", 0.0),
                lm.get("request_work_seconds", 0.0)
            )
            if not accepted:
                self.add_event(
                    "DUPLICATE_IGNORED",
                    f"Gradient dupliqué/tardif ignoré pour {r.get('task_id')} "
                    f"(déjà comptabilisé, probablement après réattribution)",
                )
                continue

            self.ps.assimilate(
                r["grad"], r["params_version"], r.get("n_samples", 1),
                self.cfg.model.lr, self.cfg.train.staleness_max)

            device = lm.get("client_device", client_id)

            self.add_event(
                "TASK_DONE",
                f"{device} termine {r['task_id']} en {lm.get('duration_seconds', 0):.2f}s"
            )

            self.add_event(
                "GRADIENT",
                f"Gradient appliqué pour {r['task_id']}"
            )
            #lm = r.get("local_metrics") or {}
            if "local_accuracy" in lm:
                self._epoch_local_acc.append(lm["local_accuracy"])
            if "avg_turns" in lm:
                self._epoch_local_turns.append(lm["avg_turns"])
        self._maybe_finalize()


    def record_client_comm_metrics(self, client_id, task_ids, report_seconds, tasks_count):
        self.sched.add_report_communication(client_id, report_seconds, tasks_count)

        self.add_event(
            "COMMUNICATION",
            f"{client_id} POST /report = {float(report_seconds or 0):.4f}s pour {tasks_count} tâche(s)"
        )
    # ----- finalisation d'epoque ----- #
    def _maybe_finalize(self):
        done, total = self.sched.progress()
        threshold = math.ceil(self.cfg.train.completion_fraction * total)
        with self.lock:
            if self._finalizing or self.finished or total == 0 or done < threshold:
                return
            self._finalizing = True

        theta, version = self.ps.get_theta()
        metrics = self.job.evaluate(theta, self.cfg.server.validation_samples)
        bw = self.ps.bandwidth_stats()
        sched_stats = self.sched.stats()
        elapsed = ((self.end_time or time.time()) - self.start_time) if self.start_time else 0
        total_compute = sched_stats.get("total_compute_seconds", 0)
        workers = max(1, sched_stats.get("active_workers", 1))
        speedup = total_compute / elapsed if elapsed > 0 else 0
        efficiency = speedup / workers if workers > 0 else 0
        throughput_gradients = bw.get("gradients_appliques", 0) / elapsed if elapsed > 0 else 0
        rec = {
            "epoch": self.epoch,
            "accuracy": metrics["accuracy"],
            "top3": metrics.get("top3"),
            "macro_f1": metrics.get("macro_f1"),
            "avg_turns": metrics.get("avg_turns"),
            "elapsed": elapsed,
            "epoch_time": time.time() - self.epoch_start,
            "version": version,
            "reassigned": sched_stats["reassigned"],
            "grads_applied": bw["gradients_appliques"],
            "grads_dropped": bw["gradients_perimes_rejetes"],
            "grads_anomalies": bw.get("gradients_anomalies_rejetes", 0),
            "active_workers": sched_stats.get("active_workers", 0),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 3),
            "throughput_gradients_per_second": round(throughput_gradients, 4),
        }
        self.history.append(rec)
        self.phase_history.setdefault(self.current_phase, []).append({
            "run_id": self.run_id,
            "phase": self.current_phase,
            "final_accuracy": rec["accuracy"],
            "final_top3": rec.get("top3"),
            "final_macro_f1": rec.get("macro_f1"),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 3),
            "throughput_gradients_per_second": round(throughput_gradients, 4),
            "active_workers": sched_stats.get("active_workers", 0),
            "rejection_rate": round((bw.get("gradients_perimes_rejetes", 0) + bw.get("gradients_anomalies_rejetes", 0)) / max(1, (bw.get("gradients_appliques", 0) + bw.get("gradients_perimes_rejetes", 0) + bw.get("gradients_anomalies_rejetes", 0))), 4),
            "epoch": self.epoch,
            "elapsed": elapsed,
        })

        #if metrics["accuracy"] >= self.cfg.train.target_accuracy or \
         #       self.epoch >= self.cfg.train.max_epochs:
         #   self.finished = True

         # Pour les expériences comparatives, on ne s'arrête pas dès que la précision cible est atteinte.
         # Cela permet de comparer 1 volontaire vs plusieurs volontaires sur la même quantité de travail.
        if self.epoch >= self.cfg.train.max_epochs:
            self.finished = True
            self.end_time = time.time()
        else:
            self.epsilon = max(self.cfg.model.epsilon_end,
                               self.epsilon * self.cfg.model.epsilon_decay)
            self._begin_epoch(self.epoch + 1)

    # ----- etat (tableau de bord) ----- #
    def status(self):
        done, total = self.sched.progress()
        last = self.history[-1] if self.history else {}
        sched_stats = self.sched.stats()
        elapsed = ((self.end_time or time.time()) - self.start_time) if self.start_time else 0
        total_compute = sched_stats.get("total_compute_seconds", 0)
        workers = max(1, sched_stats.get("active_workers", 1))

        speedup = total_compute / elapsed if elapsed > 0 else 0
        efficiency = speedup / workers if workers > 0 else 0

        bandwidth = self.ps.bandwidth_stats()
        gradients = bandwidth.get("gradients_appliques", 0)
        throughput_gradients = gradients / elapsed if elapsed > 0 else 0        
        return {
            "job": self.job.name,
            "n_params": self.job.n_params(),
            "run_id": self.run_id,
            "current_phase": self.current_phase,
            "phase_history": self.phase_history,
            "epoch": self.epoch,
            "max_epochs": self.cfg.train.max_epochs,
            "finished": self.finished,
            "progress": {"done": done, "total": total},
            "target_accuracy": self.cfg.train.target_accuracy,
            "last_accuracy": last.get("accuracy"),
            "last_top3": last.get("top3"),
            "last_f1": last.get("macro_f1"),
            "elapsed": elapsed,
            "bandwidth": bandwidth,
            "scheduler": sched_stats,
            "distributed_metrics": {
                "total_compute_seconds": round(total_compute, 3),
                "speedup": round(speedup, 3),
                "efficiency": round(efficiency, 3),
                "throughput_gradients_per_second": round(throughput_gradients, 4),
            },
            "events": self.events[-100:],
            "history": self.history,
        }
