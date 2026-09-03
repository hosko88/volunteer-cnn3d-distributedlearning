# Présentation d’état d’avancement — 05 août 2026

> **Note (mise à jour)** : le pipeline principal du système utilise désormais exclusivement l’**apprentissage distribué par SGD (parameter server)**. Les mentions « FedAvg » dans d’anciens documents sont historiques / pédagogiques.


## Diapositive 1 — Titre

Using Volunteer Computing for Distributed Learning with Convolutional Neural Networks: A Progressive Approach Toward 3D Imaging

- Mémoires de fin d’études / projet de recherche
- Approche orientée calcul volontaire, apprentissage distribué et CNN 3D
- Progression en 3 phases : CIFAR-10 → ModelNet40 → ShapeNet

---

## Diapositive 2 — Problème scientifique

Comment exploiter des machines hétérogènes et volontairement disponibles pour entraîner un modèle de deep learning, sans cluster coûteux ni infrastructure centralisée lourde ?

Objectif :
- montrer une preuve de faisabilité de calcul volontaire pour l’apprentissage distribué ;
- faire évoluer le système d’un démonstrateur 2D vers une logique 3D progressive ;
- intégrer une architecture de coordination BOINC-like avec agrégation de type FedAvg.

---

## Diapositive 3 — Contribution principale

Le projet construit un prototype de système distribué où :
- un serveur central orchestre les tâches ;
- des volontaires exécutent localement des sous-tâches ;
- les mises à jour sont agrégées à faible coût de coordination ;
- le système est conçu pour supporter des clients hétérogènes, avec tolérance aux pannes.

Contribution originale :
- progression expérimentale en trois phases ;
- démonstrateur de CNN 3D avec mécanisme d’attention ;
- cadre générique réutilisable pour d’autres tâches de la même famille.

---

## Diapositive 4 — Architecture du système

- Serveur de coordination : gestion de l’état global, distribution des tâches, agrégation, tableau de bord.
- Volontaires : machines physiques, émettant localement des mises à jour de modèle.
- Framework générique : scheduler, work generator, parameter server, compression, orchestration.
- Jobs : démonstration CNN 3D, FedAvg, attention spatiale-temporelle.

Idée clé :
- on envoie des poids locaux plutôt qu’un simple gradient brut ;
- le serveur applique une agrégation de type FedAvg.

---

## Diapositive 5 — Progression du design expérimental

### Phase 1 — CIFAR-10 / CNN 2D
- validation du paradigme de calcul distribué ;
- point de départ pour la stabilité de la boucle serveur-volontaire.

### Phase 2 — ModelNet40 / CNN 3D statique
- transition vers le domaine volumétrique 3D ;
- passage au traitement d’entrées 3D plus coûteuses en mémoire et en calcul.

### Phase 3 — ShapeNet / CNN 3D spatio-temporel avec attention
- cœur de démonstration du mémoire ;
- mécanisme d’attention pour mettre en valeur les zones les plus informatives du volume.

---

## Diapositive 6 — État d’avancement concret

### Ce qui est déjà stabilisé
- le cœur de démonstration est le job CNN 3D d’attention ;
- le serveur détient la sélection de phase et l’historique multi-phases ;
- le tableau de bord s’appuie sur un historique côté serveur ;
- le système est cohérent avec un scénario de mémoire orienté 3D progressif.

### Ce qui reste à finaliser
- un dernier nettoyage textuel documentaire ;
- la preuve de cohérence finale sur les références de recherche et la présentation orale.

---

## Diapositive 7 — Articles récents à mobiliser (2023–2026)

1. Articles récents sur l’apprentissage fédéré (Federated Learning) à l’échelle d’appareils hétérogènes :
   - 2024–2025 : travaux sur l’optimisation des communications et de l’agrégation dans des environnements hétérogènes.
   - Intérêt pour votre mémoire : ces travaux soutiennent la logique des statégies d’agrégation, de tolérance aux pannes et d’adaptation à des clients différents.

2. Articles sur l’imagerie 3D et les CNN 3D dans des contextes volumétriques :
   - 2023–2025 : CNN 3D, attention, méthodes de traitement volumétrique et analyse spatio-temporelle.
   - Intérêt pour votre mémoire : ils justifient l’usage d’un réseau 3D progressif et l’ajout d’un mécanisme d’attention pour la sélection des zones les plus utiles.

3. Travaux sur le calcul volontaire / edge computing / distributed ML à petite échelle :
   - 2023–2026 : documents sur l’usage de ressources hétérogènes, partage de calcul, et apprentissage collaboratif en environnement contraint.
   - Intérêt pour votre mémoire : ils fournissent le fondement conceptuel à un prototype BOINC-like, sans avoir besoin d’un supercalculateur.

### Article à appliquer directement sur cet environnement

Le document le plus directement exploitable pour votre cadre est un article de référence sur l’apprentissage fédéré robuste à l’hétérogénéité des clients, car votre environnement reproduit précisément cette situation :
- clients de capacités différentes ;
- transmission locale de mises à jour ;
- agrégation de poids par centrale ;
- besoin de stabilité face à des clients qui peuvent se déconnecter.

C’est ce mécanisme qui doit être présenté comme la pierre angulaire de votre démonstration expérimentale.

---

## Diapositive 8 — Justification méthodologique

Le mémoire se place dans une logique de démonstration scientifique et de prototypage :
- pas de cluster de calcul coûteux ;
- démonstration du principe de calcul volontaire ;
- insertion progressive des contraintes 3D ;
- simple cadre de validation technologique et de pertinence conceptuelle.

Cette approche est utile car elle permet de construire une preuve de faisabilité simple, compréhensible et reproductible.

---

## Diapositive 9 — Validation / preuve actuelle

Le système est validé au niveau de sa cohérence d’architecture et de son cœur de démonstration :
- le job principal correspond à la logique CNN 3D + attention ;
- le framework se prête bien à une structure fédérée ;
- l’historique et le tableau de bord sont découplés d’un stockage navigateur local ;
- la trajectoire du projet reste centrée sur le thème 3D progressif.

---

## Diapositive 10 — Conclusion et prochaines étapes

### Conclusion
Le travail démontre une architecture de calcul volontaire cohérente, applicable à un scénario de deep learning distribué, avec un passage progressif vers la 3D.

### Prochaine étape
- finaliser les derniers éléments de documentation ;
- préparer une présentation orale claire et rigoureuse ;
- aligner la bibliographie sur les références 2023–2026 les plus proches du problème.

---

## Texte de message à envoyer à l’encadrant

Bonjour,

Je vous présente l’état d’avancement de mon mémoire portant sur le calcul volontaire appliqué à l’apprentissage distribué avec des CNN 3D. La contribution principale consiste à démontrer un prototype de système BOINC-like où des volontaires hétérogènes exécutent des sous-tâches localement, puis renvoient des mises à jour de modèles qui sont agrégées au niveau du serveur selon une logique de type FedAvg.

Le projet est structuré selon une progression expérimentale en trois phases : CIFAR-10, ModelNet40 et ShapeNet. Le cœur actuel du démonstrateur repose sur un job CNN 3D avec mécanisme d’attention spatiale-temporelle, ce qui permet d’orienter le mémoire vers une logique plus proche des usages volumétriques 3D.

Au niveau de l’état de réalisation, le cadre d’orchestration, la logique de coordination et l’interface de suivi sont déjà bien structurés. Il reste à finaliser quelques éléments de documentation et de cohérence textuelle avant la présentation orale.

Je reste à votre disposition pour tout complément.

Cordialement,

[Votre nom]
