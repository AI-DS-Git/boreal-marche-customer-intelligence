import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results = pd.read_csv("docs/head_a_spot_check_results.csv")

models = [
    "Dummy baseline",
    "Logistic Regression",
    "Random Forest"
]

data = [
    results.loc[results["model"] == m, "roc_auc"].tolist()
    for m in models
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    data,
    tick_labels=models
)

plt.ylabel("ROC-AUC")
plt.title("Head A — Chronological Spot-Check ROC-AUC by Fold")
plt.grid(axis="y", alpha=0.25)

Path("docs").mkdir(exist_ok=True)

plt.tight_layout()

plt.savefig(
    "docs/head_a_roc_auc_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: docs/head_a_roc_auc_boxplot.png")
