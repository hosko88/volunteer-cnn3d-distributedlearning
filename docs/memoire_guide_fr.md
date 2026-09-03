# Guide de mémoire : Using Volunteer Computing for Distributed Learning with Convolutional Neural Networks: A Progressive Approach Toward 3D Imaging

> **Note (mise à jour)** : le pipeline principal du système utilise désormais exclusivement l’**apprentissage distribué par SGD (parameter server)**. Les mentions « FedAvg » dans d’anciens documents sont historiques / pédagogiques.


## 1. Choix du thème
Votre mémoire peut s’articuler autour de la question suivante :

> Comment utiliser des machines hétérogènes et volontairement disponibles pour entraîner un modèle de deep learning basé sur des réseaux convolutionnels 3D, dans une progression vers l’imagerie médicale 3D, sans disposer d’un cluster coûteux ?

Cette idée est intéressante parce qu’elle relie trois dimensions :
- l’optimisation distribuée,
- la faisabilité pratique sur des appareils ordinaires,
- l’applicabilité des CNN 3D à l’imagerie médicale en progression vers une analyse volumique 3D.

## 2. Structure recommandée du mémoire
### 2.1 Introduction
Présentez le contexte :
- l’essor de l’IA en imagerie médicale,
- le coût élevé des ressources de calcul,
- l’intérêt du calcul volontaire.

Expliquez pourquoi votre travail est pertinent.

### 2.2 Problématique
Formulez la question centrale :
- Peut-on entraîner un modèle 3D avec plusieurs machines physiques modestes ?
- Le calcul distribué peut-il remplacer un GPU coûteux dans un cadre académique ou expérimental ?

### 2.3 État de l’art
Faites une revue courte sur :
- l’apprentissage distribué,
- le calcul volontaire,
- les CNN 3D appliqués à l’imagerie médicale,
- l’optimisation par agrégation de poids et gradients distribués.

### 2.4 Méthodologie
Décrivez ici votre architecture :
- un serveur central,
- plusieurs clients volontaires,
- transmission de poids ou de gradients locaux,
- orchestration des sous-tâches pour l’entraînement distribué d’un CNN 3D.

C’est le cœur de votre mémoire.

### 2.5 Expérimentation
Présentez :
- le protocole de test sur machines physiques,
- les appareils utilisés,
- le nombre de volontaires,
- les métriques observées.

### 2.6 Résultats et discussion
Analysez :
- la convergence du modèle,
- la stabilité du système,
- l’impact de la latence et de la puissance des machines,
- la tolérance aux pannes.

### 2.7 Conclusion
Rédigez la conclusion en répondant à votre problématique et en soulignant les limites et perspectives.

## 3. Comment utiliser ce code pour votre mémoire
### 3.1 Sur la machine serveur
Exécutez :

```bash
python scripts/run_real_deployment.py --mode server --host 0.0.0.0 --port 5000
```

Le serveur expose :
- l’API de coordination,
- le tableau de bord,
- la distribution des tâches.

### 3.2 Sur chaque machine volontaire
Exécutez :

```bash
python scripts/run_real_deployment.py --mode volunteer --server http://IP_DU_SERVEUR:5000 --device laptop
```

Vous pouvez ajuster :
- --power pour augmenter la charge par volontaire,
- --slowdown pour simuler un appareil lent.

## 4. Ce que vous devez rédiger dans votre mémoire
### 4.1 Partie technique
Expliquez :
- le rôle du serveur,
- le rôle du volontaire,
- le mécanisme de transmission des mises à jour locales,
- la logique d’ordonnancement,
- la tolérance aux pannes,
- le cadre d’entraînement distribué pour les CNN 3D.

### 4.2 Partie expérimentale
Expliquez votre protocole :
1. installer le code sur le serveur,
2. lancer le serveur,
3. installer le code sur chaque machine volontaire,
4. lancer au moins deux ou trois volontaires,
5. mesurer la précision et le temps.

### 4.3 Partie scientifique
Mettez en avant :
- la faisabilité technique,
- l’intérêt du calcul distribué,
- les limites liées au réseau et à l’hétérogénéité.

## 5. Idées de résultats à montrer
Vous pouvez montrer dans votre mémoire :
- une courbe de précision en fonction des époques,
- un graphique du temps d’exécution avec plusieurs volontaires,
- une comparaison entre calcul centralisé et distribué,
- un commentaire sur la robustesse face aux déconnexions.

## 6. Conseils pour la rédaction
- Restez concret : montrez ce que vous avez réellement testé.
- Évitez les affirmations trop théoriques sans preuve expérimentale.
- Illustrez avec des captures d’écran du tableau de bord et des logs.
- Mettez en avant les limites du système dans la discussion.
