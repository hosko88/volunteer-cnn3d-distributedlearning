# Guide complet : comprendre le projet et faire les tests

> **Note (mise à jour)** : le pipeline principal du système utilise désormais exclusivement l’**apprentissage distribué par SGD (parameter server)**. Les mentions « FedAvg » dans d’anciens documents sont historiques / pédagogiques.


## 1. Objectif du projet

Ce projet montre comment plusieurs machines ordinaires peuvent collaborer pour entraîner un modèle d’intelligence artificielle de manière distribuée. Le cadre est conçu pour être utilisé dans un contexte de calcul volontaire, où chaque machine participante apporte une petite partie du calcul.

Le système est organisé autour de trois grandes idées :

1. un serveur central qui coordonne le travail,
2. plusieurs volontaires qui exécutent des sous-tâches,
3. une logique de fusion des résultats pour obtenir un modèle global.

---

## 2. Ce que le projet fait concrètement

Le projet fonctionne en plusieurs étapes :

1. Le serveur démarre et attend des volontaires.
2. Chaque volontaire se connecte au serveur.
3. Le serveur attribue une sous-tâche à chaque volontaire.
4. Chaque volontaire calcule localement une mise à jour du modèle.
5. Le serveur récupère les résultats et les fusionne.
6. Le modèle global est mis à jour progressivement.

Autrement dit, chaque machine n’effectue pas tout le travail seule. Chaque machine fait une petite contribution, puis le système réunit tous les résultats.

---

## 3. Les grandes parties du projet

### 3.1 Le serveur
Le serveur est la pièce centrale du système. Il a plusieurs rôles :

- initialiser le modèle global,
- créer et distribuer les tâches,
- recevoir les mises à jour locales,
- agréger les résultats,
- contrôler la progression du travail.

### 3.2 Les volontaires
Les volontaires sont les machines participantes. Chacune d’elles :

- se connecte au serveur,
- récupère la configuration de travail,
- exécute une sous-tâche locale,
- renvoie la mise à jour au serveur.

### 3.3 Le modèle
Le modèle représente l’algorithme d’apprentissage utilisé. Dans ce projet, il s’agit d’un modèle de démonstration basé sur un CNN 3D avec un mécanisme d’attention spatiale et temporelle.

### 3.4 Les tests
Les tests servent à vérifier que les composants de base fonctionnent correctement. Ils permettent de détecter rapidement les erreurs.

---

## 4. Comprendre le dossier principal du projet

Voici les dossiers essentiels à connaître :

- [jobs](jobs) : contient les algorithmes d’apprentissage.
- [framework](framework) : contient la logique de coordination et d’ordonnancement.
- [client](client) : contient le code du volontaire.
- [server](server) : contient le serveur web et le tableau de bord.
- [scripts](scripts) : contient les scripts de lancement.
- [tests](tests) : contient les tests unitaires.

---

## 5. Comprendre les fichiers importants

### 5.1 [config.py](config.py)
Ce fichier contient les réglages du système :

- les hyperparamètres du modèle,
- le nombre d’époques,
- les paramètres du transport des mises à jour,
- les réglages généraux du projet.

Il faut le considérer comme le “centre de configuration”.

### 5.2 [framework/job.py](framework/job.py)
C’est l’interface de base des jobs d’apprentissage. Tous les algorithmes doivent respecter cette structure pour être compatibles avec le système distribué.

### 5.3 [framework/coordinator.py](framework/coordinator.py)
C’est le gestionnaire principal. Il orchestre les interactions entre le serveur, les volontaires et le modèle.

### 5.4 [client/volunteer.py](client/volunteer.py)
C’est le programme exécuté par chaque volontaire. Il gère la boucle de travail :

- connexion au serveur,
- récupération de la tâche,
- calcul local,
- envoi du résultat.

### 5.5 [server/app.py](server/app.py)
C’est l’application serveur. Elle fournit les routes utilisées par les volontaires et permet l’affichage du tableau de bord.

### 5.6 [scripts/run_real_deployment.py](scripts/run_real_deployment.py)
C’est le script à utiliser pour lancer le système sur de vraies machines physiques.

---

## 6. Comprendre le mécanisme de contribution originale

La contribution originale repose sur deux éléments importants :

1. l’apprentissage distribué avec agrégation de poids,
2. l’attention spatiale et temporelle intégrée au modèle CNN 3D.

### 6.1 Agrégation de poids
Au lieu de transmettre uniquement des gradients, le système transmet des mises à jour locales de poids. Chaque volontaire apprend localement puis envoie ses poids au serveur.

Le serveur combine ensuite ces poids avec une stratégie de fusion. Cela correspond à une logique proche de l’apprentissage fédéré.

### 6.2 Attention spatiale et temporelle
Le mécanisme d’attention sert à aider le modèle à se focaliser sur les informations les plus pertinentes. Dans un contexte 3D, cela peut aider à mettre l’accent sur certaines régions du volume ou certaines caractéristiques plus importantes que d’autres.

C’est bien cette idée qui constitue l’apport original du travail.

---

## 7. Comment le code fonctionne en pratique

### 7.1 Étape 1 : le serveur initialise le modèle
Le serveur crée un modèle de départ. C’est le point de départ de l’apprentissage.

### 7.2 Étape 2 : le serveur crée des tâches
Le serveur découpe le travail en sous-tâches. Chaque sous-tâche contient une petite partie du problème à résoudre.

### 7.3 Étape 3 : le volontaire récupère une tâche
Le volontaire contacte le serveur, reçoit une tâche, puis effectue un calcul local.

### 7.4 Étape 4 : le volontaire envoie sa mise à jour
Il renvoie au serveur la mise à jour locale du modèle.

### 7.5 Étape 5 : le serveur agrège
Le serveur fusionne cette mise à jour au modèle global.

### 7.6 Étape 6 : le système continue
Le processus se répète jusqu’à ce que l’entraînement atteigne la fin définie.

---

## 8. Comprendre les jobs

Le terme job correspond à un type d’algorithme ou à un scénario d’apprentissage.

### 8.1 Job de base
Le job de base sert à tester la structure du framework.

### 8.2 Job CNN 3D
Ce job constitue un exemple de modèle d’apprentissage distribué basé sur un CNN 3D.

### 8.3 Job attention-federated
Ce job ajoute l’attention spatiale et temporelle au modèle. C’est celui qui est le plus cohérent avec votre thème de mémoire.

---

## 9. Comment exécuter le projet

### 9.1 Prérequis
Il faut avoir Python installé. Si vous êtes sur Windows, il peut être utile d’utiliser l’invocation via py ou uv.

### 9.2 Installation des dépendances
Dans le dossier du projet :

```bash
uv pip install -r requirements-server.txt
```

ou, si vous n’utilisez pas uv :

```bash
pip install -r requirements-server.txt
```

### 9.3 Lancer le serveur
```bash
python scripts/run_real_deployment.py --mode server --host 0.0.0.0 --port 5000
```

### 9.4 Lancer un volontaire
Dans un autre terminal :

```bash
python scripts/run_real_deployment.py --mode volunteer --server http://IP_DU_SERVEUR:5000 --device laptop
```

---

## 10. Comment faire les tests

### 10.1 Tester un job simple
Le plus simple est de tester un job directement depuis Python.

Exemple :

```bash
python -c "from jobs.cnn_3d.attention_fedavg_job import AttentionFedAvgCNN3DJob; job = AttentionFedAvgCNN3DJob(); params = job.init_params(); task = job.make_task(0,0,1,0.1); local, n, metrics = job.compute_local_update(params, task); print(local.shape, n, metrics)"
```

### 10.2 Tester un fichier de test
Si vous avez un test déjà écrit, vous pouvez le lancer avec :

```bash
uv run pytest -q tests/test_attention_fedavg_job.py
```

### 10.3 Tester tous les tests
```bash
uv run pytest -q
```

---

## 11. Comprendre les tests existants

Les tests servent à vérifier que les fonctions de base fonctionnent.

Un test typique vérifie :

- si un modèle peut être initialisé,
- si une tâche peut être créée,
- si une mise à jour locale peut être calculée,
- si la fusion de poids fonctionne.

---

## 12. Comment lire un test

Un test est souvent écrit sous cette forme :

```python
def test_xxx():
    job = AttentionFedAvgCNN3DJob()
    params = job.init_params()
    task = job.make_task(0, 0, 1, 0.1)
    local, n, metrics = job.compute_local_update(params, task)
    assert local.shape == params.shape
```

Cela veut dire :

- on crée un job,
- on initialise un modèle,
- on crée une tâche,
- on exécute le calcul local,
- puis on vérifie que le résultat a la bonne forme.

---

## 13. Ce qu’il faut vérifier si un test échoue

Si un test échoue, il faut regarder trois choses :

1. l’import du module,
2. l’initialisation des paramètres,
3. la logique de calcul local ou d’agrégation.

Les erreurs les plus fréquentes sont :

- module introuvable,
- mauvaise importation de chemin,
- erreur dans une fonction de calcul,
- type de données incompatible.

---

## 14. Conseils pratiques pour bien comprendre le système

### 14.1 Commencez par le job
Le plus simple est de lire d’abord le fichier du job, car c’est là que le cœur de l’apprentissage est défini.

### 14.2 Ensuite lisez le coordinateur
Le coordinateur vous aidera à comprendre la logique globale.

### 14.3 Puis le volontaire
Le volontaire montre comment une machine participe réellement au système.

### 14.4 Enfin, faites un test simple
Un test simple vous aidera à valider votre compréhension.

---

## 15. Ce que vous devez retenir pour votre mémoire

Pour votre mémoire, vous pouvez expliquer que le système :

- distribue l’apprentissage sur plusieurs machines,
- exploite du calcul volontaire,
- utilise un modèle CNN 3D,
- intègre un mécanisme d’attention spatiale et temporelle,
- et agrège les mises à jour locales de façon dynamique.

C’est précisément cette combinaison qui constitue l’intérêt du travail.

---

## 16. Résumé rapide

Si vous voulez résumer tout le projet en une phrase :

> Ce système montre comment plusieurs machines peuvent collaborer pour entraîner un modèle intelligent de façon distribuée, avec un mécanisme d’attention et une logique d’agrégation de poids.

---

## 17. Commandes utiles à garder

```bash
python scripts/run_real_deployment.py --mode server --host 0.0.0.0 --port 5000
python scripts/run_real_deployment.py --mode volunteer --server http://IP_DU_SERVEUR:5000 --device laptop
uv run pytest -q
```

---

## 18. Si vous voulez aller plus loin

Vous pouvez ensuite ajouter :

- des vrais jeux de données 3D,
- des métriques plus sérieuses,
- des graphiques de convergence,
- et un protocole expérimental complet pour votre mémoire.
