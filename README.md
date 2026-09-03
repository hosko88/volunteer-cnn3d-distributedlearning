# Using Volunteer Computing for Distributed Learning with Convolutional Neural Networks: A Progressive Approach Toward 3D Imaging

Application Python de type **BOINC** conçue pour démontrer qu’on peut exploiter
la **puissance de calcul simultanée de smartphones, de PC Linux et de PC Windows**
pour réaliser un **apprentissage distribué de modèles CNN 3D** dans un cadre de
**calcul volontaire**, sans super-calculateur coûteux ni énergivore.

La contribution originale du système repose sur un **mécanisme d’attention
spatiale et temporelle** intégré au modèle CNN 3D afin de mettre en valeur les
zones les plus informatives du volume, tout en stabilisant l’agrégation entre
volontaires au fil d’une progression expérimentale en **trois phases** :

1. Phase 1 — CIFAR-10 / CNN 2D
2. Phase 2 — ModelNet40 / CNN 3D statique
3. Phase 3 — ShapeNet / CNN 3D spatio-temporel avec attention

Le cœur de démonstration du projet est le **job CNN 3D d’attention**.
La suite de l’infrastructure est volontairement générique : le même serveur,
la même orchestration BOINC et la même API de dashboard peuvent être réutilisés
pour un autre algorithme de la même famille sans changements structurels.

---

## 1. Idée et architecture

Modèle de calcul : **apprentissage distribué (SGD / parameter server)** avec agrégation de gradients.

```
   Serveur (coordination)                         Volontaires (calcul)
   ┌───────────────────────────┐                  ┌────────────────────┐
   │ paramètres globaux θ       │  ── θ (poids) ─▶ │ smartphone Android │
   │ optimiseur Adam            │                  │ PC Linux           │
   │ ordonnanceur BOINC         │ ◀─ gradients ──  │ PC Windows         │
   │ tolérance aux pannes       │   (compressés)   └────────────────────┘
   │ tableau de bord temps réel │
   └───────────────────────────┘
```

* Le serveur détient les paramètres globaux θ. Il découpe le travail en
  **sous-tâches**, les distribue **selon la puissance** de chaque volontaire,
  **réattribue** les sous-tâches non rendues (tolérance aux pannes), reçoit les
  **gradients** calculés par chaque volontaire et applique une **mise à jour SGD/Adam**.
* Chaque volontaire télécharge θ, entraîne localement un modèle CNN 3D sur un lot
  local, puis renvoie ses **poids mis à jour**. Il ne transmet donc pas un gradient,
  mais un état de modèle local.
* **On transmet des gradients compressés**, pas les poids complets. Le serveur
  applique immédiatement la mise à jour (Adam). Ce paradigme est particulièrement
  adapté au calcul volontaire : asynchrone, tolérant aux pannes, et économe en
  bande passante.

Pourquoi des gradients ? Cette approche est adaptée au calcul volontaire et à
l’apprentissage distribué : communication légère, asynchronisme naturel, et
réattribution simple des sous-tâches en cas de déconnexion.

---

## 2. Arborescence (où placer chaque fichier)

```
volunteer_rl/
├── README.md
├── requirements.txt
├── config.py                  ← tous les hyperparamètres (un seul endroit)
│
├── framework/                 ← LE "BOINC" générique (agnostique à l'algorithme)
│   ├── __init__.py
│   ├── job.py                 ← interface TrainingJob (point d'extension)
│   ├── compression.py         ← (dé)compression des gradients fp16 + top-k
│   ├── parameter_server.py    ← agrège les gradients + Adam + comptabilité réseau
│   ├── work_generator.py      ← fabrique les sous-tâches d'une époque
│   ├── scheduler.py           ← ordonnanceur : puissance, timeouts, fiabilité
│   └── coordinator.py         ← cycle des époques, finalisation, signal de fin
│
├── jobs/                      ← les ALGORITHMES branchables
│   ├── __init__.py
│   └── cnn_3d/                ← cœur de démonstration du thème (CNN 3D + attention)
│       ├── __init__.py
│       ├── attention.py       ← attention spatiale-temporelle
│       ├── job.py             ← CNN3DJob de base
│       ├── job.py             ← CNN3DJob de base
│       └── attention_job.py   ← job principal (CNN 3D + attention, SGD distribué)
│
├── server/
│   ├── __init__.py
│   ├── app.py                 ← API HTTP (Flask)
│   └── dashboard.html         ← tableau de bord temps réel
│
├── client/
│   ├── __init__.py
│   ├── device_info.py         ← détection OS / cœurs / RAM / Android, puissance
│   └── volunteer.py           ← client volontaire (boucle de gradient)
│
├── data/
│   └── (fichiers de démonstration de la phase 3D, à adapter localement)
│
└── scripts/                   ← points d'entrée
    ├── prepare_data.py        ← vérifie/affiche la base
    ├── run_server.py          ← lance le serveur (déploiement réel)
    ├── run_volunteer.py       ← lance un volontaire (Android/Windows/Linux)
    ├── run_simulation.py      ← simulation locale de bout en bout (flotte simulée)
    ├── run_sequential.py      ← référence séquentielle (1 appareil)
    └── analyze.py             ← toutes les métriques + figures du mémoire
```

---

## 3. Installation

Sur le serveur (et pour l'analyse) :

```bash
pip install -r requirements.txt
```

Sur un **volontaire**, le strict minimum suffit :

```bash
pip install numpy requests
```

---

## 4. Démarrage rapide (sur une seule machine)

```bash
python scripts/run_server.py --job attention
python scripts/run_simulation.py      # serveur + flotte hétérogène simulée, de bout en bout
```

## 4.1 Déploiement réel sur machines physiques

Pour tester entre plusieurs ordinateurs du réseau local :

Sur la machine serveur :
```bash
python scripts/run_real_deployment.py --mode server --host 0.0.0.0 --port 5000
```

Sur chaque machine volontaire :
```bash
python scripts/run_real_deployment.py --mode volunteer --server http://IP_DU_SERVEUR:5000 --device laptop
```

Vous pouvez ensuite ouvrir l’adresse du serveur dans votre navigateur pour visualiser le tableau de bord.

Un guide détaillé pour votre mémoire est disponible dans [docs/memoire_guide_fr.md](docs/memoire_guide_fr.md).

Pour observer la **tolérance aux pannes** (un smartphone se déconnecte en cours) :

```bash
python scripts/run_simulation.py --kill
```

Pour un transport **encore plus léger** (sparsification des gradients) :

```bash
python scripts/run_simulation.py --topk 0.10
```

À la fin, toutes les métriques sont écrites dans `results/distributed_run.json`.

---

## 5. Déploiement réel (plusieurs appareils)

### 5.1 Lancer le serveur (sur la machine de coordination)

```bash
python scripts/run_server.py --host 0.0.0.0
```

Au démarrage, le serveur **affiche son adresse locale** (ex. `http://192.168.1.10:5000`)
et la **commande à donner aux volontaires**. Le **tableau de bord temps réel** est
à cette même adresse dans un navigateur. À la fin du calcul, le serveur **signale**
la convergence avec les métriques finales.

### 5.2 Participer depuis un volontaire

Remplacez `IP_DU_SERVEUR` par l'adresse affichée par le serveur.

**Android (Termux)**
```bash
pkg install python
pip install numpy requests
python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device smartphone
```

**Windows (invite de commande)**
```bat
pip install numpy requests
python scripts\run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device windows-pc
```

**Linux (terminal)**
```bash
pip install numpy requests
python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device linux-pc
```

Options utiles : `--power N` (nombre de sous-tâches par requête, sinon automatique),
`--slowdown S` (ralentissement artificiel par sous-tâche, pour simuler un appareil lent).

> Les volontaires n'ont **rien de spécial** à gérer en cas de coupure : si un
> appareil se déconnecte, le serveur **réattribue** automatiquement ses sous-tâches.

---

## 6. Obtenir les métriques et figures du mémoire

```bash
python scripts/run_sequential.py --epochs 16   # référence séquentielle (courbes complètes)
python scripts/run_simulation.py               # exécution distribuée
python scripts/analyze.py                       # génère les 7 figures + metrics_memoire.json
```

Figures produites dans `results/` :

| Fichier | Démontre |
|---|---|
| `fig1_convergence_qualite.png` | qualité de la progression de la phase : précision/top-3/F1 montent au fil des époques |
| `fig2_plus_value.png` | **plus-value vs séquentiel** : cible atteinte ~K× plus vite avec K volontaires |
| `fig3_passage_echelle.png` | temps pour atteindre la cible vs **nombre de volontaires** |
| `fig4_bande_passante.png` | **allègement du transport** : gradients compressés vs poids bruts |
| `fig5_heterogeneite.png` | **répartition de la charge** selon la puissance de chaque appareil |
| `fig6_tolerance_pannes.png` | **convergence malgré** la déconnexion de volontaires |
| `fig7_cout_accessibilite.png` | **coût et énergie** : flotte volontaire vs GPU cloud vs serveur dédié |

---

## 7. Extensibilité : faire tourner un autre algorithme après le mémoire

Le `framework/` est agnostique par conception. Pour un nouvel algorithme, créez
`jobs/mon_algo/job.py` avec une classe héritant de `TrainingJob` (`framework/job.py`)
et implémentez :

```python
class MonJob(TrainingJob):
    name = "Mon algorithme"
    def init_params(self): ...            # vecteur de paramètres initial
    def n_params(self): ...               # taille du vecteur
    def init_opt_state(self): ...         # état initial de l'optimiseur (serveur)
    def make_task(self, epoch, index, seed, epsilon): ...   # décrit une sous-tâche
    def compute_gradient(self, params, task): ...           # côté volontaire -> gradient
    def apply_gradient(self, params, grad, opt_state, lr): ...# côté serveur -> pas d'optimiseur
    def evaluate(self, params, n): ...    # métriques (au moins 'accuracy')
```

Puis branchez-le dans `scripts/run_server.py` (à la place du job actif de démonstration).
L'ordonnancement, la tolérance aux pannes, le transport par gradients compressés,
l'hétérogénéité et le tableau de bord fonctionnent **sans aucune modification**.

---

## 8. Note d'honnêteté sur les mesures

La machine de développement peut ne disposer que d'**un seul cœur**. Le gain de
temps **parallèle** n'y est donc pas directement observable (les fils se partagent
le CPU). En conséquence :

* sont **mesurés réellement** : la convergence, la qualité de la phase, la
  tolérance aux pannes, la répartition selon l'hétérogénéité, et la réduction de
  bande passante ;
* est **modélisé** (à partir du temps par gradient réellement mesuré) : le gain de
  temps parallèle de `fig2`/`fig3`. Il se matérialise sur un **vrai déploiement
  multi-appareils**.

Le **modèle de coût** (`fig7`) repose sur des hypothèses de prix et de puissance
clairement indiquées et **éditables** en tête de `scripts/analyze.py` — à remplacer
par vos chiffres locaux. Le message est le **rapport** entre options (la flotte ne
paie que l'électricité marginale d'appareils déjà possédés, capital ≈ 0), invariant
d'échelle.
