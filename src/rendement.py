
import mlflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")

runs = mlflow.search_runs(experiment_names=["tp4-automl"])

auto = runs[runs["params.outil"].isin(["flaml", "sklearn-maison"])].copy()
auto["budget"] = auto["params.budget_s"].astype(int)
auto = auto.drop_duplicates(["params.outil", "budget"]).sort_values("budget")

refs = runs[runs["params.outil"] == "scikit-learn"].drop_duplicates(
    "tags.mlflow.runName"
)

plt.figure(figsize=(8, 5))

for outil, groupe in auto.groupby("params.outil"):
    plt.plot(
        groupe["budget"],
        groupe["metrics.f1"],
        "o-",
        label=f"AutoML ({outil})"
    )
    for budget, f1 in zip(groupe["budget"], groupe["metrics.f1"]):
        print(f"{outil:15s} budget {budget:4d} s -> F1 {f1:.4f}")

for _, run in refs.iterrows():
    nom = run["tags.mlflow.runName"]
    f1 = run["metrics.f1"]
    plt.axhline(f1, linestyle="--", linewidth=1)
    plt.text(
        auto["budget"].min(),
        f1 + 0.003,
        f"{nom} ({f1:.4f})",
        fontsize=8
    )
    print(f"{nom:22s} F1 {f1:.4f}")

plt.xscale("log")
budgets = sorted(auto["budget"].unique())
plt.xticks(budgets, [str(b) for b in budgets])
plt.minorticks_off()
plt.xlabel("Budget de calcul (secondes, echelle log)")
plt.ylabel("F1 sur le jeu de test")
plt.title("Rendement du budget de calcul")
plt.legend(loc="lower right")
plt.savefig("reports/rendement.png", dpi=120, bbox_inches="tight")

print("Graphique ecrit : reports/rendement.png")
