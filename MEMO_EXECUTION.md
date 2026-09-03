# MÉMO D'EXÉCUTION
## Using Volunteer Computing for Distributed Learning with Convolutional Neural Networks: A Progressive Approach Toward 3D Imaging

**Système de calcul volontaire pour l'apprentissage distribué de CNN 3D**  
Auteur du framework : adapté pour le thème ci-dessus  
Date : août 2026

> ⚠️ **ÉTAT ACTUEL (voir PROGRESSION.md pour le détail à jour)** : ce mémo a été
> écrit avant l'ajout de la progression en 3 phases et de la double piste
> heavy/light. Il reste valable pour les principes généraux (parameter server,
> compression, tolérance aux pannes) mais deux points sont dépassés :
> - Il n'y a plus un seul job de démonstration `AttentionCNN3DJob` : le
>   serveur lance désormais `python scripts/run_server.py --phase {1,2,3}`
>   (CIFAR-10/CNN2D, ModelNet40/CNN3D, ModelNet40/CNN3D+Attention), avec
>   `AttentionCNN3DJob` conservé uniquement comme piste "light" (NumPy, sans
>   PyTorch) pour les volontaires sans torch (ex. smartphones), activée par
>   défaut et désactivable via `--no-light-track`.
> - L'installation côté volontaire est `pip install -r requirements-client.txt`
>   (torch CPU + numpy + requests pour la piste heavy), pas seulement
>   `numpy requests` comme indiqué plus bas.
>
> En cas de doute, PROGRESSION.md et le code (`scripts/run_server.py`,
> `client/volunteer.py`) font foi.

---

## 1. Présentation rapide du système

Ce système permet d'entraîner un **CNN 3D avec mécanisme d'attention spatiale/temporelle** de manière **distribuée** sur des machines volontaires hétérogènes (PC Windows, PC Linux, smartphones Android via Termux).

Architecture :
- **Serveur de coordination** (machine principale) : détient les paramètres globaux θ, distribue les sous-tâches, agrège les mises à jour (SGD distribué (parameter server)), tolère les pannes.
- **Volontaires** : téléchargent θ, calculent un gradient local (ou mise à jour de poids) sur un mini-lot, renvoient le résultat compressé (fp16 + top-k).

Progression expérimentale prévue :
1. **Phase 1** : CIFAR-10 volumeisé → CNN 2D/3D de base (démonstration actuelle)
2. **Phase 2** : ModelNet40 → CNN 3D statique
3. **Phase 3** : ShapeNet / volumes médicaux → CNN 3D spatio-temporel + attention

Le job principal de démonstration est `AttentionCNN3DJob`.

---

## 2. Prérequis communs

- Python 3.9 ou supérieur (recommandé 3.10+)
- Accès réseau local entre le serveur et les volontaires (même Wi-Fi / LAN)
- Pare-feu : autoriser le port 5000 (TCP) en entrée sur la machine serveur

### Installation minimale côté serveur
```bash
pip install -r requirements-server.txt
```

### Installation minimale côté volontaire (Windows ou Linux)
```bash
pip install numpy requests
```
(ou `pip install -r requirements-client.txt`)

---

## 3. Préparation des données (une seule fois, côté serveur)

```bash
python scripts/prepare_data.py
```
Cela télécharge automatiquement CIFAR-10 (Phase 1) dans le dossier `data/`.

---

## 4. Lancement du serveur (machine de coordination)

### Sur Linux
```bash
cd /chemin/vers/volunteer_system
python scripts/run_server.py --host 0.0.0.0 --port 5000 --job attention
```

### Sur Windows (Invite de commandes ou PowerShell)
```cmd
cd C:\chemin\vers\volunteer_system
python scripts\run_server.py --host 0.0.0.0 --port 5000 --job attention
```

Le serveur affiche :
- Son adresse IP locale (ex. `http://192.168.1.25:5000`)
- La commande exacte à donner aux volontaires
- Un tableau de bord en temps réel accessible dans le navigateur à cette adresse

**Important** : laissez cette fenêtre ouverte pendant toute la durée de l'expérience.

---

## 5. Lancement des volontaires

Remplacez `IP_DU_SERVEUR` par l'adresse affichée par le serveur (ex. `192.168.1.25`).

### 5.1 Volontaire sous Linux
```bash
cd /chemin/vers/volunteer_system
python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device linux-pc
```

Options utiles :
```bash
# Forcer une puissance (nombre de sous-tâches demandées à chaque tour)
python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device linux-pc --power 4

# Simuler un appareil plus lent
python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device linux-pc --slowdown 0.5
```

### 5.2 Volontaire sous Windows
```cmd
cd C:\chemin\vers\volunteer_system
python scripts\run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device windows-pc
```

Avec options :
```cmd
python scripts\run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device windows-pc --power 4
python scripts\run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device windows-pc --slowdown 0.5
```

### 5.3 Plusieurs volontaires sur la même machine (simulation locale)
Ouvrez plusieurs terminaux et lancez plusieurs instances de `run_volunteer.py` avec des `--device` différents (ex. `pc1`, `pc2`, `smartphone`).

---

## 6. Déploiement réel simplifié (recommandé)

Un script unifié existe :

**Serveur :**
```bash
python scripts/run_real_deployment.py --mode server --host 0.0.0.0 --port 5000 --job attention
```

**Volontaire (Linux/Windows) :**
```bash
python scripts/run_real_deployment.py --mode volunteer --server http://IP_DU_SERVEUR:5000 --device mon-pc
```

---

## 7. Simulation locale complète (pour tests et génération de métriques)

Sur une seule machine (utile pour valider avant déploiement multi-machines) :

```bash
# Serveur + flotte hétérogène simulée
python scripts/run_simulation.py

# Avec tolérance aux pannes (un volontaire "meurt")
python scripts/run_simulation.py --kill

# Avec compression forte
python scripts/run_simulation.py --topk 0.25
```

Référence séquentielle (1 seul appareil) :
```bash
python scripts/run_sequential.py --epochs 30
```

Analyse et figures pour le mémoire :
```bash
python scripts/analyze.py
```
Les figures et métriques sont générées dans `results/`.

---

## 8. Tableau de bord

Ouvrez dans un navigateur :
```
http://IP_DU_SERVEUR:5000
```
Vous y verrez en temps réel :
- Nombre de volontaires connectés
- Progression des époques
- Précision / perte
- Répartition de la charge
- État des sous-tâches

---

## 9. Arrêt et résultats

- Le serveur s'arrête automatiquement lorsqu'il atteint `target_accuracy` ou `max_epochs`.
- Les résultats sont écrits dans `results/distributed_run.json`.
- Les volontaires peuvent être arrêtés proprement avec `Ctrl+C`.

---

## 10bis. Compression des gradients (version optimisée)

Le module `framework/compression.py` a été renforcé :

| Technique | Gain typique | Description |
|-----------|--------------|-------------|
| **int8** (nouveau) | x4 vs fp32 | Quantification avec échelle dynamique |
| **fp16** | x2 | Quantification classique |
| **top-k** | x2 à x10 | Ne transmet que les k% plus grands gradients |
| **zlib niveau 9** | +10-20% | Meilleure compression lossless |
| **Error Feedback** | qualité ↑ | Conserve l'erreur pour stabiliser la convergence |
| **indices uint16** | léger | Empaquetage plus serré pour petits modèles |

**Réglage par défaut** (dans `config.py`) :
```python
dtype = "int8"
topk  = 0.25
```
→ réduction typique **x5 à x11** par rapport aux gradients fp32 bruts, tout en gardant une bonne convergence grâce à l'Error Feedback côté volontaire.

Pour tester d'autres réglages :
```bash
# Dans config.py, ou via argument simulation
python scripts/run_simulation.py --topk 0.10
```


## 10. Conseils de performance pour un système très efficace

1. **Réseau** : privilégiez le Wi-Fi 5 GHz ou le câble Ethernet pour le serveur.
2. **Nombre de volontaires** : 3 à 8 machines donnent déjà un excellent speedup.
3. **Compression** : `topk=0.5` + `fp16` (valeur par défaut dans `config.py`) offre un excellent compromis.
4. **Timeout** : `task_timeout=120` permet de gérer les appareils qui s'endorment.
5. **Puissance** : laissez le client estimer automatiquement (`--power` uniquement si vous voulez forcer).
6. **Pare-feu Windows** : autorisez Python sur le port 5000 lors du premier lancement.
7. **Antivirus** : temporairement désactiver l'analyse en temps réel peut accélérer le calcul NumPy.

---

## 11. Structure des fichiers importants

```
volunteer_system/
├── config.py                  ← tous les hyperparamètres
├── MEMO_EXECUTION.md          ← ce fichier
├── README.md
├── requirements-server.txt
├── requirements-client.txt
├── framework/                 ← moteur BOINC-like (ne pas modifier pour le mémoire)
├── jobs/cnn_3d/               ← modèle CNN 3D + attention (cœur scientifique)
├── client/                    ← client volontaire (Windows / Linux / Android)
├── server/                    ← API Flask + dashboard
├── scripts/                   ← points d'entrée
└── docs/                      ← guides mémoire et présentation
```

---

## 12. Dépannage rapide

| Problème | Solution |
|----------|----------|
| Volontaire ne se connecte pas | Vérifier IP, pare-feu, même réseau |
| "serveur injoignable" | Lancer le serveur avec `--host 0.0.0.0` |
| Calcul très lent | Vérifier que NumPy est bien installé (version ≥ 1.26) |
| Port déjà utilisé | Changer `--port 5001` |
| Erreur de données | Relancer `python scripts/prepare_data.py` |

---

**Bon courage pour votre mémoire !**  
Ce système est conçu pour être à la fois pédagogique, réaliste et performant en conditions de calcul volontaire hétérogène.
