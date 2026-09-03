"""
Serveur de parametres
=====================
Detient les parametres globaux theta et l'etat de l'optimiseur Adam. Recoit des
GRADIENTS compresses de la part des volontaires, les applique (SGD asynchrone
avec gestion de la peremption / staleness), et sert theta aux volontaires.

Comptabilise aussi la bande passante (brute vs compressee) pour DEMONTRER le gain
du transport par gradients compresses.
"""

import threading
import numpy as np

from .compression import encode_vector, decode_vector, raw_size_bytes


class ParameterServer:
    def __init__(self, job, transport_cfg):
        self.job = job
        self.tcfg = transport_cfg
        self.theta = job.init_params().astype(np.float32)
        self.opt = job.init_opt_state()
        self.version = 0
        self.lock = threading.Lock()
        # comptabilite reseau
        self.bytes_up = 0          # volontaire -> serveur (gradients compresses)
        self.bytes_down = 0        # serveur -> volontaire (parametres compresses)
        self.raw_up_fp32 = 0       # ce qu'auraient coute des gradients/poids bruts fp32
        self.n_grads_applied = 0
        self.n_grads_dropped = 0   # gradients rejects a cause de staleness / peremption
        self.n_grads_anomalies = 0 # gradients aberrants rejects comme anomalies Byzantines simples
        self.grad_clip = 1.0
        self.max_grad_norm = 50.0  # seuil plus realiste (avant 1e4 laissait passer presque tout)
        self.base_lr = None        # fixe au premier assimilate si besoin
        self.warmup_steps = 50     # warmup lineaire pour stabiliser le debut

    # ----- service des parametres (telechargement par le volontaire) ----- #
    def params_payload(self):
        with self.lock:
            payload, nbytes, _ = encode_vector(self.theta, dtype=self.tcfg.dtype, topk=1.0)
            v = self.version
        self.bytes_down += nbytes
        return payload, v

    def get_theta(self):
        with self.lock:
            return self.theta.copy(), self.version

    # ----- assimilation d'un gradient (televersement par le volontaire) ----- #
    def assimilate(self, grad_payload, params_version, n_samples, lr, staleness_max):
        grad = decode_vector(grad_payload)

        # Detection d'anomalie AVANT clipping : le clipping doit stabiliser
        # des gradients legitimes mais bruites, pas masquer un gradient
        # aberrant (client corrompu/malveillant/panne) en l'ecrasant a une
        # norme acceptable avant meme la verification. L'ordre inverse
        # rendait ce garde-fou totalement inoperant (toute valeur, meme
        # extreme, ressortait avec une norme <= grad_clip apres l'avoir
        # "corrigee" silencieusement).
        raw_gnorm = float(np.linalg.norm(grad, ord=2))
        is_anomalous = (not np.isfinite(raw_gnorm)) or (raw_gnorm > self.max_grad_norm)

        # Clipping cote serveur (securite + stabilite) -- uniquement sur les
        # gradients qui passent la verification d'anomalie.
        max_norm = getattr(self, 'grad_clip', 1.0)
        if not is_anomalous and max_norm and raw_gnorm > max_norm and raw_gnorm > 0:
            grad = (grad * (max_norm / raw_gnorm)).astype(np.float32)
        grad_norm = raw_gnorm
        comp_bytes = len(grad_payload)  # taille reçue (approx base64) -> comptee ci-dessous proprement
        # taille reellement transferee (octets compresses) : recalculee par l'appelant idealement ;
        # ici on estime via la longueur du payload base64 -> octets
        comp_bytes = int(len(grad_payload) * 3 / 4)
        raw = raw_size_bytes(self.job.n_params(), "fp32")

        with self.lock:
            staleness = self.version - params_version

            # Filtre simple et robuste pour les gradients aberrants (clients Byzantins ou
            # malveillants) : on ne cherche pas ici un modele DP, mais un garde-fou
            # numerique qui rejette les mises a jour trop grandes pour le schema actuel.
            # Cette mesure est volontairement conservative pour ne pas casser les clients
            # honorables, tout en laissant un compteur separé de robustesse dans le dashboard.
            if is_anomalous:
                self.n_grads_anomalies += 1
                self.bytes_up += comp_bytes
                self.raw_up_fp32 += raw
                return False, staleness

            if staleness > staleness_max:
                # peremption non bornee : on rejette (le travail sera refait sur theta a jour)
                self.n_grads_dropped += 1
                self.bytes_up += comp_bytes
                self.raw_up_fp32 += raw
                return False, staleness

            # --- Learning rate schedule optimise ---
            # 1) Warmup lineaire sur les premiers pas (evite explosions au demarrage)
            # 2) Decroissance douce 1/sqrt(t) apres warmup (bonne pratique SGD distribue)
            # 3) Attenuation selon la staleness (Hogwild! / DistBelief)
            step = self.n_grads_applied + 1
            if step <= self.warmup_steps:
                lr_sched = lr * (step / max(1, self.warmup_steps))
            else:
                # plateau puis decroissance lente
                lr_sched = lr / (1.0 + 0.05 * ((step - self.warmup_steps) ** 0.5))

            lr_eff = lr_sched / (1.0 + 0.3 * max(0, staleness))

            self.theta, self.opt = self.job.apply_gradient(
                self.theta, grad, self.opt, lr_eff
            )
            self.version += 1
            self.n_grads_applied += 1
            self.bytes_up += comp_bytes
            self.raw_up_fp32 += raw
        return True, staleness

    # ----- statistiques reseau ----- #
    def bandwidth_stats(self):
        total = self.bytes_up + self.bytes_down
        raw_equiv = self.raw_up_fp32 * 2  # aller (poids) + retour (gradients) en fp32 brut
        return {
            "octets_televerses_gradients": self.bytes_up,
            "octets_telecharges_parametres": self.bytes_down,
            "octets_total_compresse": total,
            "octets_equivalent_brut_fp32": raw_equiv,
            "facteur_reduction": (raw_equiv / total) if total else 0.0,
            "gradients_appliques": self.n_grads_applied,
            "gradients_perimes_rejetes": self.n_grads_dropped,
            "gradients_anomalies_rejetes": self.n_grads_anomalies,
        }
