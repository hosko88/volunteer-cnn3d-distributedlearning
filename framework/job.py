"""
Interface des jobs d'entrainement (extensibilite)
=================================================
Le "BOINC" (framework/) est AGNOSTIQUE a l'algorithme. Pour faire tourner un
nouvel algorithme apres le memoire, il suffit d'ecrire une classe qui herite de
`TrainingJob` et d'implementer ces quelques methodes. Le serveur, l'ordonnanceur,
la tolerance aux pannes, le transport par gradients et le tableau de bord
fonctionnent alors sans modification.

Modele de calcul = SGD distribue avec serveur de parametres :
  * le serveur detient les parametres globaux theta + l'etat de l'optimiseur ;
  * chaque volontaire recoit theta, calcule un GRADIENT sur un lot local, le renvoie ;
  * le serveur agrege les gradients et applique un pas d'optimiseur.

C'est exactement le schema "d'agregation (implicite) de gradients" du code de
reference C++ -- mais distribue sur des machines heterogenes via le reseau.
"""

from abc import ABC, abstractmethod


class TrainingJob(ABC):
    """Contrat minimal d'un algorithme distribuable sur le BOINC."""

    #: nom lisible de l'algorithme (affiche par le serveur / le tableau de bord)
    name: str = "job"

    # ----- cote SERVEUR : initialisation ----- #
    @abstractmethod
    def init_params(self):
        """Retourne le vecteur de parametres initial (numpy 1D, float32)."""

    @abstractmethod
    def n_params(self) -> int:
        """Nombre total de parametres (taille du vecteur)."""

    # ----- cote SERVEUR : fabrication d'une sous-tache ----- #
    @abstractmethod
    def make_task(self, epoch: int, index: int, seed: int, epsilon: float) -> dict:
        """
        Decrit une sous-tache (workunit) serialisable en JSON : tout ce dont un
        volontaire a besoin pour produire un gradient (graine, taille du lot, ...).
        """

    def enrich_task_with_data(self, task: dict) -> dict:
        """
        Côté SERVEUR uniquement.
        Échantillonne le mini-batch correspondant au seed de la tâche et
        l'encode (float16 + npz + base64) pour transmission.
        Le volontaire ne télécharge JAMAIS le dataset ni une partition :
        il reçoit uniquement le mini-batch nécessaire à cette tâche.
        Implémentation par défaut : no-op (rétrocompatibilité).
        """
        return task

    # ----- cote VOLONTAIRE : calcul du gradient ----- #
    @abstractmethod
    def compute_gradient(self, params_flat, task: dict):
        """
        Coeur du calcul deporte. A partir des parametres globaux et de la
        description de tache, retourne :
            grad_flat (numpy 1D), n_echantillons (int), metriques_locales (dict)
        AUCUN etat d'optimiseur ici : le volontaire ne calcule qu'un gradient brut.
        Si task contient "batch_payload", utiliser ces données (chemin principal).
        Sinon, fallback sur sample_batch (tests / rétrocompatibilité).
        """

    # ----- cote SERVEUR : application du gradient agrege ----- #
    @abstractmethod
    def apply_gradient(self, params_flat, grad_flat, opt_state, lr: float):
        """
        Applique un pas d'optimiseur (p.ex. Adam) au vecteur global.
        Retourne (nouveaux_params, nouvel_etat_optimiseur).
        L'etat de l'optimiseur vit cote serveur (les volontaires en sont dispenses).
        """

    @abstractmethod
    def init_opt_state(self):
        """Etat initial de l'optimiseur cote serveur (p.ex. moments d'Adam)."""

    # ----- cote SERVEUR : evaluation ----- #
    @abstractmethod
    def evaluate(self, params_flat, n: int) -> dict:
        """Evalue le modele global. Retourne un dict de metriques (au moins 'accuracy')."""
