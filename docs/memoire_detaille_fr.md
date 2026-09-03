# Mémoire détaillé : présentation du projet de calcul volontaire pour l’apprentissage distribué avec CNN 3D

## 1. Introduction

Ce projet est un prototype de système informatique qui montre comment plusieurs machines ordinaires, comme des ordinateurs de bureau, des portables ou des téléphones, peuvent travailler ensemble pour entraîner un modèle d’intelligence artificielle basé sur des réseaux convolutionnels 3D.

L’idée principale est simple : au lieu d’utiliser un seul ordinateur très puissant, on répartit le travail entre plusieurs appareils. Chacun d’eux reçoit une petite partie du travail à effectuer, calcule une réponse à partir d’un sous-volume ou d’un lot local, puis envoie le résultat au serveur principal.

Ce type d’approche s’appelle le calcul volontaire, parce que les appareils participent volontairement à une tâche commune.

Le projet présenté ici est une version simplifiée et pédagogique de ce principe. Il permet de comprendre comment un système distribué peut être construit, comment les mises à jour locales sont envoyées, comment les tâches sont réparties et comment le résultat final est obtenu dans une progression vers l’imagerie 3D.

---

## 2. Ce que fait ce code en une phrase

Ce code permet de :

- lancer un serveur central,
- distribuer des sous-tâches à plusieurs volontaires,
- faire calculer des mises à jour locales sur chaque machine,
- récupérer les résultats,
- les agréger pour améliorer progressivement un modèle convolutionnel 3D,
- afficher l’évolution du système dans un tableau de bord.

En termes simples, il sert à montrer qu’on peut faire de l’apprentissage automatique de façon distribuée, sans disposer d’un superordinateur, dans un cadre orienté vers l’imagerie 3D.

---

## 3. Le problème que ce projet veut résoudre

L’apprentissage automatique moderne consomme beaucoup de puissance de calcul. En particulier, les réseaux de neurones profonds, surtout les CNN 3D, demandent beaucoup de temps et de mémoire.

Or, toutes les machines ne sont pas utilisées de façon optimale. Un ordinateur personnel, un portable ou un téléphone possède souvent des ressources inutilisées. Le projet montre qu’il est possible d’utiliser cette puissance disponible pour faire du calcul collaboratif au service de l’apprentissage distribué pour l’imagerie 3D.

L’intérêt principal est double :

1. réduire les coûts de calcul ;
2. rendre l’apprentissage possible avec des équipements plus modestes.

---

## 4. L’idée générale du système

Le système comporte trois grandes parties :

1. un serveur principal,
2. plusieurs volontaires,
3. un mécanisme de coordination.

### 4.1 Le serveur principal
Le serveur central est le chef d’orchestre. Il :

- reçoit les demandes des volontaires,
- attribue des sous-tâches,
- collecte les résultats,
- fusionne les informations,
- met à jour le modèle global.

### 4.2 Les volontaires
Les volontaires sont les machines qui participent. Chacune d’elles :

- reçoit les paramètres du modèle,
- calcule une petite partie du travail,
- renvoie un résultat au serveur.

### 4.3 La coordination
Le système doit savoir :

- qui participe,
- quelles tâches sont attribuées,
- quelles tâches sont en retard,
- comment gérer une machine qui se déconnecte.

C’est cette coordination qui rend le système robuste.

---

## 5. Les concepts clés à comprendre

### 5.1 Le modèle
Un modèle est une représentation mathématique qui apprend à partir de données. Dans ce projet, il s’agit d’un modèle simple de type CNN 3D, conçu comme une petite intelligence artificielle capable d’apprendre à partir d’exemples volumétriques dans une perspective d’imagerie 3D.

### 5.2 Le gradient et les mises à jour locales
Le gradient est une information mathématique qui indique dans quelle direction il faut modifier les paramètres du modèle pour améliorer ses performances.

Dans ce projet, chaque volontaire calcule une mise à jour locale à partir de la petite tâche qui lui a été attribuée. Cette mise à jour peut être exprimée sous forme de gradient ou de poids localement recalculés, selon la stratégie d’agrégation retenue.

### 5.3 La distribution du travail
Au lieu de demander à une seule machine de tout faire, on découpe le travail en sous-tâches. Chaque machine se voit attribuer une partie du calcul destinée à l’apprentissage d’un modèle CNN 3D en contexte distribué.

### 5.4 Le calcul volontaire
Le calcul volontaire consiste à utiliser des machines physiques, parfois hétérogènes, pour exécuter des tâches distribuées dans un cadre d’apprentissage collaboratif orienté vers les modèles 3D.

---

## 6. Architecture du projet

Le projet est organisé en plusieurs dossiers et modules. Voici la logique générale.

### 6.1 Le dossier framework
Ce dossier contient la partie générique du système. C’est la base technique du calcul distribué.

Il comprend :

- un contrat d’interface pour les tâches d’apprentissage,
- un serveur de paramètres,
- un générateur de tâches,
- un ordonnanceur,
- un coordinateur.

### 6.2 Le dossier jobs
Ce dossier contient les algorithmes concrets à exécuter.

Dans ce projet, on trouve :

- un job générique de démonstration basé sur un CNN 3D,
- un job de plugin basé sur l’apprentissage par renforcement, compatible avec l’architecture générale.

### 6.3 Le dossier server
Ce dossier contient tout ce qui concerne le serveur web de coordination et le tableau de bord.

### 6.4 Le dossier client
Ce dossier contient le programme qui tourne sur chaque machine volontaire.

### 6.5 Le dossier scripts
Ce dossier contient les scripts de lancement, de simulation et de déploiement réel.

---

## 7. Description détaillée des fichiers principaux

### 7.1 config.py
Ce fichier contient tous les paramètres du système. On y trouve :

- les paramètres du modèle,
- les réglages de l’entraînement,
- la taille des lots,
- le nombre d’époques,
- les seuils d’arrêt,
- les réglages du transport des gradients.

Il est important parce qu’il centralise les options du projet.

### 7.2 framework/job.py
C’est l’interface de base pour définir un travail d’apprentissage distribuée.

Tout nouveau modèle doit respecter cette structure. C’est le point d’entrée pour ajouter un nouveau type de tâche.

### 7.3 framework/coordinator.py
C’est le cœur du système. Il orchestre les étapes suivantes :

- création des tâches,
- attribution aux volontaires,
- réception des résultats,
- application des gradients,
- mise à jour du modèle global.

### 7.4 framework/parameter_server.py
Il sert de serveur de paramètres. Il enregistre l’état actuel du modèle et applique les mises à jour issues des gradients.

### 7.5 framework/scheduler.py
Cet élément décide qui reçoit quelle tâche. Il prend en compte la capacité des machines, leur vitesse et leur fiabilité.

### 7.6 framework/work_generator.py
Il crée les sous-tâches qui seront distribuées.

### 7.7 server/app.py
C’est l’application serveur. Elle permet à la machine coordinateur de répartir le travail entre volontaires et de fournir un tableau de bord.

### 7.8 client/volunteer.py
C’est le programme exécuté par chaque volontaire. Il :

- se connecte au serveur,
- récupère les paramètres du modèle,
- demande une sous-tâche,
- calcule un gradient,
- envoie le résultat au serveur.

### 7.9 scripts/run_real_deployment.py
C’est le point d’entrée recommandé pour tester le système sur plusieurs machines physiques.

### 7.10 jobs/cnn_3d/job.py
C’est un exemple de job de démonstration basé sur un CNN 3D. Il a été ajouté pour rendre le projet plus proche d’un usage en imagerie médicale 3D, avec une logique de progression vers le traitement volumétrique 3D.

---

## 8. Le fonctionnement pas à pas

Voici maintenant le fonctionnement complet du système, de manière très simple.

### Étape 1 : le serveur démarre
Le serveur principal se lance. Il attend que des volontaires se connectent.

### Étape 2 : les volontaires se connectent
Chaque machine volontaire envoie ses informations de base :

- le nom de l’appareil,
- le système d’exploitation,
- le nombre de cœurs,
- la mémoire disponible,
- une estimation de puissance.

### Étape 3 : le serveur attribue des tâches
Le serveur découpe le travail en sous-tâches. Chaque sous-tâche est envoyée à une machine volontaire.

### Étape 4 : le volontaire calcule
La machine volontaire reçoit les paramètres du modèle et la sous-tâche. Elle calcule ensuite un gradient local.

### Étape 5 : le résultat est envoyé au serveur
Le volontaire renvoie le gradient au serveur. Ce gradient sert à améliorer le modèle global.

### Étape 6 : le modèle est mis à jour
Le serveur combine les gradients reçus et modifie les paramètres du modèle.

### Étape 7 : le cycle continue
Le processus continue jusqu’à ce que le système considère que la convergence est suffisante ou que le nombre d’époques défini soit atteint.

---

## 9. Ce que fait le job CNN 3D

Le job CNN 3D est une version simplifiée d’un problème d’apprentissage profond. Il sert de démonstrateur pédagogique pour la perspective de recherche visée par le thème.

Ce job ne travaille pas sur des données volumineuses réelles, mais il simule le comportement d’un modèle de type réseau convolutif 3D.

La contribution originale du système réside dans l’intégration d’un **mécanisme d’attention spatiale et temporelle**. Ce mécanisme permet au modèle de mettre l’accent sur les régions les plus pertinentes du volume et d’améliorer la stabilité de l’apprentissage lorsque plusieurs volontaires participent à la mise à jour du modèle global.

Son rôle est de montrer que :

- un modèle peut être entraîné de manière distribuée,
- des machines différentes peuvent envoyer des mises à jour locales,
- le système peut servir de base expérimentale pour une mémoire sur le calcul volontaire et l’imagerie 3D.

Il est donc utile pour illustrer la démarche scientifique, même s’il ne remplace pas un vrai modèle de classification d’images 3D sur de vraies données médicales.

---

## 10. Comment installer le projet

Avant de lancer le code, il faut installer les dépendances Python.

### 10.1 Sous Windows
Ouvrez un terminal dans le dossier du projet puis exécutez :

```bash
pip install -r requirements-server.txt
```

Si vous voulez seulement un usage simple avec un volontaire, vous pouvez aussi installer :

```bash
pip install numpy requests flask psutil matplotlib
```

### 10.2 Sous Linux ou Mac
Même logique :

```bash
pip install -r requirements-server.txt
```

---

## 11. Comment exécuter le projet sur une seule machine

C’est la manière la plus simple pour tester le système.

### 11.1 Lancer le serveur
Dans le dossier du projet :

```bash
python scripts/run_real_deployment.py --mode server --host 127.0.0.1 --port 5000
```

### 11.2 Lancer un volontaire
Dans un second terminal :

```bash
python scripts/run_real_deployment.py --mode volunteer --server http://127.0.0.1:5000 --device laptop
```

Le système commence alors à travailler.

---

## 12. Comment exécuter le projet sur plusieurs machines physiques

C’est la version la plus intéressante pour votre mémoire, car elle montre l’usage réel du calcul volontaire.

### 12.1 Sur la machine serveur
Exécutez :

```bash
python scripts/run_real_deployment.py --mode server --host 0.0.0.0 --port 5000
```

Cette machine devient le coordinateur du réseau.

### 12.2 Sur chaque machine volontaire
Exécutez :

```bash
python scripts/run_real_deployment.py --mode volunteer --server http://IP_DU_SERVEUR:5000 --device laptop
```

Remplacez IP_DU_SERVEUR par l’adresse IP réelle de la machine serveur.

### 12.3 Ouvrir le tableau de bord
Dans un navigateur, ouvrez :

```text
http://IP_DU_SERVEUR:5000/
```

Vous pourrez observer les informations en temps réel.

---

## 13. Comment tester le système

Le test peut être fait de trois façons.

### 13.1 Test de base avec une seule machine
C’est le test le plus simple. On lance un serveur et un volontaire sur la même machine.

### 13.2 Test distribué sur plusieurs machines
On utilise une vraie machine serveur et plusieurs machines volontaires. C’est le meilleur test pour votre mémoire, car il reflète le vrai contexte du calcul volontaire.

### 13.3 Test de robustesse
On peut simuler une déconnexion d’un volontaire pour voir si le système continue malgré la panne.

---

## 14. Comment savoir que tout fonctionne

Voici les signes d’un bon fonctionnement :

- le serveur démarre sans erreur,
- les volontaires se connectent,
- des tâches sont attribuées,
- les résultats sont envoyés au serveur,
- le tableau de bord se met à jour,
- les métriques évoluent au fil du temps.

Si tout se passe bien, vous verrez des messages de connexion et des informations de progression dans le terminal.

---

## 15. Ce que le tableau de bord affiche

Le tableau de bord permet de visualiser en temps réel plusieurs informations :

- le nombre d’époques,
- la précision du modèle,
- le nombre de tâches en cours,
- le nombre de volontaires actifs,
- le temps de calcul,
- les communications réseau,
- les erreurs ou déconnexions éventuelles.

C’est un outil très utile pour expliquer le fonctionnement du système dans un mémoire ou une soutenance.

---

## 16. Ce que vous pouvez montrer dans votre mémoire

Pour faire un mémoire solide, il est recommandé de présenter plusieurs éléments concrets.

### 16.1 La problématique
Vous pouvez expliquer qu’un apprentissage profond demande beaucoup de ressources et que le calcul volontaire peut offrir une alternative économique.

### 16.2 La méthode
Vous pouvez décrire la structure du système : serveur, volontaires, tâches, gradients, coordination.

### 16.3 L’expérimentation
Vous pouvez raconter votre protocole de test :

- une machine serveur,
- deux ou trois machines volontaires,
- lancement du système,
- observation du tableau de bord,
- mesure de la progression.

### 16.4 Les résultats
Vous pouvez montrer :

- que le système fonctionne,
- que les machines participent bien,
- que le calcul est réparti,
- que le système reste opérationnel malgré les pannes.

---

## 17. Limites du projet

Il est important d’être honnête dans un mémoire. Ce projet est un prototype pédagogique, pas une plateforme de production complète.

Ses limites sont les suivantes :

- le modèle est simple,
- les données utilisées ne sont pas toujours réalistes,
- la communication réseau peut être lente,
- certains appareils peuvent se déconnecter,
- la performance dépend fortement du réseau et de la puissance des machines.

Ces limites font partie du travail scientifique. Elles montrent que le projet est réaliste, mais encore perfectible.

---

## 18. Conseils pour une bonne présentation orale ou écrite

Si vous devez présenter ce projet devant un professeur, voici quelques conseils très utiles :

1. commencez par expliquer le problème de façon simple,
2. montrez ensuite l’idée du calcul distribué,
3. décrivez le rôle du serveur et des volontaires,
4. montrez le tableau de bord,
5. expliquez les résultats obtenus,
6. parlez enfin des limites et des perspectives.

Un bon exposé doit être simple, concret et illustré par des exemples réels.

---

## 19. Résumé final

Ce projet montre qu’il est possible de construire un système d’apprentissage distribué basé sur des machines physiques et volontaires. Il permet de comprendre les bases du calcul collaboratif, de la coordination distribuée et de l’apprentissage automatique moderne.

Il est particulièrement intéressant pour un mémoire, car il mélange plusieurs notions importantes :

- intelligence artificielle,
- réseau de calcul,
- parallélisme,
- apprentissage distribué,
- systèmes embarqués et machines hétérogènes,
- expérimentation scientifique.

---

## 20. Version ultra simple pour un lecteur profane

Si vous voulez résumer le projet en une seule idée :

> Ce programme montre qu’on peut utiliser plusieurs ordinateurs ordinaires pour faire ensemble un travail de calcul difficile, au lieu d’avoir besoin d’un seul ordinateur très puissant.

C’est cela l’essentiel du projet.
