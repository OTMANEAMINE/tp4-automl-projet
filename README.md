# TP04 — AutoML et suivi des expérimentations avec MLflow

## 1. Présentation du projet

Ce projet compare des modèles de classification pour la détection de transactions frauduleuses. Il utilise des modèles de référence développés manuellement, puis FLAML pour automatiser la recherche de modèles et d'hyperparamètres. MLflow permet d'enregistrer et de comparer les expérimentations.

## 2. Technologies utilisées

- Python
- FLAML : recherche automatisée de modèles
- Scikit-learn : modèles de référence et métriques
- MLflow : suivi des expérimentations
- Pandas : manipulation des données
- Matplotlib : visualisation des résultats

## 3. Structure du projet

```text
tp4-automl-projet/
├── data/
├── reports/
│   ├── flaml_10s.log
│   ├── flaml_30s.log
│   ├── flaml_60s.log
│   ├── flaml_120s.log
│   ├── rendement.png
│   └── comparatif.md
├── src/
│   ├── references.py
│   ├── automl.py
│   ├── inspecter.py
│   ├── fuite.py
│   └── rendement.py
├── donnees.py
├── requirements.txt
└── README.md
```

## 4. Installation

Créer et activer un environnement virtuel :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install -r requirements.txt
```

## 5. Génération des données

Depuis le dossier racine du projet :

```powershell
python donnees.py
```

Cette commande génère le jeu de données utilisé par les expériences.

## 6. Lancement de MLflow

Dans un premier terminal, démarrer le serveur :

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Ouvrir ensuite l'interface dans le navigateur :

http://127.0.0.1:5000

Laisser ce terminal ouvert pendant les expériences.

## 7. Exécution des expériences

Dans un deuxième terminal, depuis la racine du projet et avec l'environnement virtuel activé :

```powershell
python src/references.py
python src/automl.py
python src/inspecter.py reports/flaml_60s.log
python src/fuite.py
python src/rendement.py
```

Les scripts permettent d'enregistrer les modèles de référence, de lancer les expériences FLAML, d'inspecter les essais, d'illustrer une fuite de données et de produire un graphique comparatif.

## 8. Résultats expérimentaux

| Approche | F1-score |
|---|---:|
| Régression logistique | 0.7407 |
| Random Forest manuel | 0.8669 |
| FLAML — 10 secondes | 0.8409 |
| FLAML — 30 secondes | 0.8582 |
| FLAML — 60 secondes | 0.8582 |
| FLAML — 120 secondes | 0.8561 |

Le Random Forest manuel obtient le meilleur F1-score parmi ces expériences. FLAML améliore son score entre 10 et 30 secondes, mais l'augmentation du budget ne garantit pas une amélioration supplémentaire.

## 9. Fuite de données

Le script `src/fuite.py` illustre le risque d'utiliser une variable qui révèle indirectement la cible. Une performance artificiellement parfaite pendant le test peut s'effondrer lorsque cette variable n'est plus disponible en production. Les variables utilisées doivent donc être disponibles au moment réel de la prédiction.

## 10. Conclusion

Ce projet met en évidence l'intérêt de l'AutoML pour explorer automatiquement plusieurs configurations et de MLflow pour suivre les expériences. Les résultats doivent être interprétés en tenant compte du jeu de données, du protocole d'évaluation et des risques de fuite de données.
