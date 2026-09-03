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
Ordonnanceur facon BOINC
========================
Distribue les sous-taches aux volontaires et assure la robustesse :

  * Distribution PROPORTIONNELLE A LA PUISSANCE : un volontaire puissant reçoit
    plusieurs sous-taches par requete (moins d'allers-retours), un smartphone une
    seule. La puissance est declaree par le volontaire (auto-benchmark).
  * TOLERANCE AUX PANNES : une sous-tache non rendue avant un delai est
    automatiquement REATTRIBUEE a un autre volontaire (le travail global ne
    s'arrete jamais).
  * Suivi de FIABILITE par volontaire (taux de sous-taches menees a terme).
"""

import time
import threading
from collections import deque


class ClientStat:
    def __init__(self, info, power):
        self.info = info
        self.power = power          # nb de sous-taches accordees par requete
        self.completed = 0
        self.timeouts = 0
        self.assigned = 0
        self.last_seen = time.time()
        self.total_compute_seconds = 0.0
        self.last_task_seconds = 0.0
        self.total_request_seconds = 0.0
        self.total_report_seconds = 0.0
        self.total_communication_seconds = 0.0
        self.total_task_seconds = 0.0
        self.avg_task_time = 0.0
        self.speed_score = 1.0
        self.dynamic_power = power

    @property
    def reliability(self):
        tot = self.completed + self.timeouts
        return self.completed / tot if tot else 1.0


class Scheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clients = {}
        self.pending = deque()
        self.inflight = {}          # task_id -> (client_id, deadline)
        self.tasks = {}             # task_id -> task
        self.completed_ids = set()  # tâches déjà comptées (évite double comptage après réattribution)
        self.done = 0
        self.total = 0
        self.reassigned = 0
        self.lock = threading.Lock()

    # ----- volontaires ----- #
    def register(self, client_id, info, power):
        with self.lock:
            if client_id not in self.clients:
                self.clients[client_id] = ClientStat(info, max(1, int(power)))
            self.clients[client_id].last_seen = time.time()

    # ----- chargement d'une epoque ----- #
    def load_epoch(self, tasks):
        with self.lock:
            self.pending.clear()
            self.inflight.clear()
            self.completed_ids.clear()
            self.tasks = {t["task_id"]: t for t in tasks}
            self.pending.extend(t["task_id"] for t in tasks)
            self.done = 0
            self.total = len(tasks)
            self.reassigned = 0

    # ----- tolerance aux pannes : reattribution des sous-taches expirees ----- #
    def _reclaim_expired(self):
        now = time.time()
        expired = [tid for tid, (cid, dl) in self.inflight.items() if now > dl]
        for tid in expired:
            cid, _ = self.inflight.pop(tid)
            if tid in self.completed_ids:
                # Déjà terminée entre-temps → ne pas réattribuer
                continue
            if cid in self.clients:
                self.clients[cid].timeouts += 1
            self.pending.appendleft(tid)        # remis en tete de file
            self.reassigned += 1

    # ----- assignation (par lot proportionnel a la puissance) ----- #
    def assign(self, client_id):
        with self.lock:
            self._reclaim_expired()

            if client_id not in self.clients:
                return []

            c = self.clients[client_id]

            # Au début, on utilise la puissance estimée par benchmark.
            # Après quelques tâches, on utilise le temps moyen réel.
            measured_clients = [
                x for x in self.clients.values()
                if x.avg_task_time > 0
            ]

            if measured_clients and c.avg_task_time > 0:
                fastest = min(x.avg_task_time for x in measured_clients)

                # Ratio de performance réel :
                # plus avg_task_time est faible, plus la machine est rapide.
                ratio = fastest / c.avg_task_time

                # Le plus rapide peut recevoir jusqu'à 8 tâches.
                n = round(8 * ratio)
                n = max(1, min(8, n))
            else:
                # Phase de démarrage : benchmark initial
                n = max(1, min(8, int(c.power)))

            c.dynamic_power = n
            c.power = n

            batch = []
            deadline = time.time() + self.cfg.train.task_timeout

            while self.pending and len(batch) < n:
                tid = self.pending.popleft()
                self.inflight[tid] = (client_id, deadline)
                c.assigned += 1
                batch.append(self.tasks[tid])

            c.last_seen = time.time()
            return batch    

    # ----- compte rendu d'une sous-tache ----- #
    def report(self, client_id, task_id, duration_seconds=0.0, request_work_seconds=0.0):
        with self.lock:
            # Déjà comptée (ex. réattribution + rapport tardif) → ignorer
            if task_id in self.completed_ids:
                return False

            if task_id in self.inflight:
                del self.inflight[task_id]
                self.completed_ids.add(task_id)
                self.done += 1

                if client_id in self.clients:
                    c = self.clients[client_id]
                    c.completed += 1

                    compute = float(duration_seconds or 0.0)
                    request_time = float(request_work_seconds or 0.0)

                    c.last_task_seconds = compute
                    c.total_compute_seconds += compute
                    
                    # Moyenne glissante du temps de calcul réel
                    if compute > 0:
                        if c.avg_task_time <= 0:
                            c.avg_task_time = compute
                        else:
                            alpha = 0.20
                            c.avg_task_time = alpha * compute + (1 - alpha) * c.avg_task_time

                    # Score relatif : 1.0 = volontaire le plus rapide observé
                    measured = [
                        x.avg_task_time for x in self.clients.values()
                        if x.avg_task_time > 0
                    ]

                    if measured and c.avg_task_time > 0:
                        fastest = min(measured)
                        c.speed_score = fastest / c.avg_task_time

                    c.total_request_seconds += request_time
                    c.total_communication_seconds += request_time
                    c.total_task_seconds += compute + request_time

                return True

            return False

    def add_report_communication(self, client_id, report_seconds=0.0, tasks_count=1):
        """
        Ajoute le temps POST /report.
        Un POST /report peut contenir plusieurs tâches, donc on l'ajoute au total
        et il sera moyenné ensuite.
        """
        with self.lock:
            if client_id not in self.clients:
                return

            c = self.clients[client_id]
            report_seconds = float(report_seconds or 0.0)

            c.total_report_seconds += report_seconds
            c.total_communication_seconds += report_seconds
            c.total_task_seconds += report_seconds
    
    # ----- etat ----- #
    def progress(self):
        with self.lock:
            return self.done, self.total

    def stats(self):
        with self.lock:
            pending_count = len(self.pending)
            inflight_count = len(self.inflight)
            done_count = self.done

            total_completed_tasks = sum(c.completed for c in self.clients.values())
            total_compute_seconds = sum(c.total_compute_seconds for c in self.clients.values())
            total_request_seconds = sum(c.total_request_seconds for c in self.clients.values())
            total_report_seconds = sum(c.total_report_seconds for c in self.clients.values())
            total_communication_seconds = sum(c.total_communication_seconds for c in self.clients.values())

            return {
                "tasks": {
                    "total": self.total,
                    "pending": pending_count,
                    "inflight": inflight_count,
                    "done": done_count,
                    "reassigned": self.reassigned,
                },
                "total_compute_seconds": round(total_compute_seconds, 3),
                "total_request_seconds": round(total_request_seconds, 3),
                "total_report_seconds": round(total_report_seconds, 3),
                "total_communication_seconds": round(total_communication_seconds, 3),
                "active_workers": len(self.clients),
                "total_completed_tasks": total_completed_tasks,
                "avg_communication_per_task": round(total_communication_seconds / total_completed_tasks, 4) if total_completed_tasks else 0,
                "clients": {
                    cid: {
                        "info": c.info,
                        "completed": c.completed,
                        "timeouts": c.timeouts,
                        "assigned": c.assigned,
                        "reliability": round(c.reliability, 3),
                        "power": c.power,
                        "last_seen": c.last_seen,
                        "avg_task_seconds": round(c.total_compute_seconds / c.completed, 3) if c.completed else 0,
                        "last_task_seconds": round(c.last_task_seconds, 3),
                        "total_compute_seconds": round(c.total_compute_seconds, 3),
                        "total_request_seconds": round(c.total_request_seconds, 3),
                        "total_report_seconds": round(c.total_report_seconds, 3),
                        "total_communication_seconds": round(c.total_communication_seconds, 3),
                        "avg_communication_seconds": round(c.total_communication_seconds / c.completed, 4) if c.completed else 0,
                        "avg_total_task_seconds": round(c.total_task_seconds / c.completed, 4) if c.completed else 0,
                        "avg_task_time_dynamic": round(c.avg_task_time, 4),
                        "speed_score": round(c.speed_score, 4),
                        "dynamic_power": c.dynamic_power,
                    }
                    for cid, c in self.clients.items()
                },
                "reassigned": self.reassigned,
            }   