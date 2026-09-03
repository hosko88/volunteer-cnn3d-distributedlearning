"""
Generateur de sous-taches (workunits)
=====================================
Fabrique, pour chaque epoque, la liste des sous-taches a distribuer. Chaque
sous-tache porte un identifiant unique et la description produite par le job
(graine, taille du lot, epsilon). Les graines sont reproductibles.
"""


class WorkGenerator:
    def __init__(self, job, cfg):
        self.job = job
        self.cfg = cfg
        self._uid = 0

    def create_epoch(self, epoch, epsilon):
        tasks = []
        for i in range(self.cfg.train.n_tasks_per_epoch):
            self._uid += 1
            seed = epoch * 100_003 + i * 7919 + self.cfg.data.seed
            t = self.job.make_task(epoch, i, seed, epsilon)
            t["task_id"] = self._uid
            tasks.append(t)
        return tasks
