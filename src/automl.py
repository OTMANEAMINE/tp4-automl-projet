
import sys
import time
from pathlib import Path

import pandas as pd
import mlflow
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Budget en secondes : 60 par defaut
budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60

# Charger les donnees
df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

# Separaration apprentissage / test
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Creer le dossier des rapports
Path("reports").mkdir(exist_ok=True)
journal = f"reports/flaml_{budget}s.log"

# Recherche AutoML
automl = AutoML()
t0 = time.time()

automl.fit(
    Xtr,
    ytr,
    task="classification",
    metric="f1",
    time_budget=budget,
    seed=42,
    verbose=0,
    log_file_name=journal,
    log_type="all"
)

duree = round(time.time() - t0, 1)
f1 = round(f1_score(yte, automl.predict(Xte)), 4)

essayees = [
    k for k, v in automl.best_config_per_estimator.items()
    if v is not None
]

print(f"Budget : {budget} s")
print(f"Famille retenue : {automl.best_estimator}")
print(f"F1 test : {f1}")
print(f"Duree : {duree} s")
print("Familles essayees :", essayees)
print("Meilleure configuration :", automl.best_config)

# Enregistrer dans MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp4-automl")

with mlflow.start_run(run_name=f"flaml-{budget}s"):
    mlflow.log_params({
        "outil": "flaml",
        "budget_s": budget,
        "famille": automl.best_estimator
    })
    mlflow.log_metrics({
        "f1": f1,
        "f1_validation": round(1 - automl.best_loss, 4),
        "duree_s": duree
    })
    mlflow.log_artifact(journal)
