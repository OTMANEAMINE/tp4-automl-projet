
# Bloc E — Comparaison des plateformes AutoML

## 1. Objectif

L'objectif est de comparer une solution AutoML locale, FLAML, avec trois plateformes cloud : Google Vertex AI, Amazon SageMaker Autopilot et Azure Automated ML.

La comparaison porte sur le coût, la facilité d'utilisation, l'automatisation, l'explicabilité, le déploiement et la réversibilité.

## 2. Tableau comparatif

| Critère | FLAML local | Google Vertex AI | Amazon SageMaker Autopilot | Azure Automated ML |
|---|---|---|---|---|
| Type | Bibliothèque Python open source | Plateforme cloud | Plateforme cloud | Plateforme cloud |
| Installation | Environnement Python local | Compte Google Cloud | Compte AWS | Compte Azure |
| Automatisation | Sélection du modèle et réglage des hyperparamètres | Entraînement AutoML selon les tâches prises en charge | Prétraitement, sélection, entraînement et réglage | Entraînement, sélection et réglage |
| Contrôle technique | Élevé | Moyen à élevé selon l'interface et les API | Moyen à élevé selon le mode utilisé | Moyen à élevé selon le mode utilisé |
| Coût d'entraînement | Pas de frais de service FLAML ; calcul local à la charge de l'utilisateur | Facturation selon les ressources et services utilisés | Facturation selon les ressources et services utilisés | Facturation selon les ressources et services utilisés |
| Déploiement géré | À mettre en place séparément | Services de déploiement disponibles | Déploiement sur endpoint disponible | Déploiement sur endpoint disponible |
| Explicabilité | Dépend du modèle et des outils ajoutés | Fonctionnalités variables selon la tâche | Rapports de performance et d'importance des variables disponibles pour certains cas | Fonctionnalités d'interprétation disponibles selon le modèle et la configuration |
| Réversibilité | Élevée : code et environnement contrôlés localement | Dépend des services utilisés et des artefacts exportés | Dépend des artefacts et services utilisés | Dépend des artefacts et services utilisés |
| Score F1 sur notre dataset | Mesuré localement | Non mesuré | Non mesuré | Non mesuré |

## 3. Résultats expérimentaux locaux

Les expériences suivantes ont été exécutées avec le dataset de transactions frauduleuses :

- Régression logistique : référence simple.
- Random Forest : modèle réglé manuellement.
- FLAML : recherches avec des budgets de 10, 30, 60 et 120 secondes.

Les scores F1 et les durées doivent être récupérés depuis MLflow pour renseigner les résultats numériques définitifs.

Les plateformes cloud n'ont pas été exécutées sur ce dataset. Aucun score F1 ne leur est attribué.

## 4. Analyse

FLAML est adapté aux expériences locales, à la reproductibilité et à la maîtrise du code. Il permet de rechercher rapidement des modèles tout en limitant le budget de calcul.

Les plateformes cloud sont intéressantes lorsque le projet nécessite une infrastructure gérée, des capacités de calcul évolutives, des outils de déploiement ou une intégration dans un environnement cloud existant.

En revanche, leur utilisation nécessite une configuration supplémentaire et peut entraîner des coûts de calcul, de stockage ou de déploiement.

## 5. Conclusion

Pour ce TP, FLAML constitue une solution pertinente pour comparer des modèles avec un budget de calcul maîtrisé et suivre les résultats dans MLflow.

Le choix d'une plateforme cloud dépendrait du contexte de production, du volume des données, des besoins de déploiement, de la gouvernance et du budget disponible.

La comparaison reste qualitative pour les trois plateformes cloud, car aucune expérience directe n'a été menée sur ces services.
