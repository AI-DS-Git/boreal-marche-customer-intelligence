"""Global interpretability for Head A: permutation importance on the held-out slice.

Importance = drop in PR-AUC (the model's primary metric) when a predictor, or a group of related predictors, is
shuffled. Group importance matters because many predictors are near-duplicates (spend, quantity and revenue per
order), so single-feature importance understates each of them. Run: python scripts/importance.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, ".")
from retrain import CURRENT_ART, DATA, ID, prepare, split  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402

OUT = Path("reports/interpretability")


def main(repeats: int = 15) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = prepare(df)
    _, _, (Xte, yte) = split(X, y, df[ID])
    model = joblib.load(CURRENT_ART)
    rng = np.random.default_rng(42)
    base = average_precision_score(yte, model.predict_proba(Xte)[:, 1])

    def drop(cols):
        d = []
        for _ in range(repeats):
            Xp = Xte.copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].iloc[idx].to_numpy()
            d.append(base - average_precision_score(yte, model.predict_proba(Xp)[:, 1]))
        return float(np.mean(d)), float(np.std(d))

    single = []
    for c in Xte.columns:
        m, s = drop([c])
        single.append(dict(feature=c, pr_auc_drop=m, sd=s))
    pd.DataFrame(single).sort_values("pr_auc_drop", ascending=False).to_csv(OUT / "permutation_importance_features.csv", index=False)
    grp = []
    for g, members in ModelService.GROUPS.items():
        cols = [m for m in members if m in Xte.columns]
        m, s = drop(cols)
        grp.append(dict(group=g, members=" ".join(cols), pr_auc_drop=m, sd=s))
    g = pd.DataFrame(grp).sort_values("pr_auc_drop", ascending=False)
    g.to_csv(OUT / "permutation_importance_groups.csv", index=False)
    print("baseline PR-AUC", round(base, 4))
    print(pd.DataFrame(single).sort_values("pr_auc_drop", ascending=False).head(10).round(4).to_string())
    print(g.round(4).to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        gg = g.iloc[::-1]
        ax.barh(gg.group, gg.pr_auc_drop, xerr=gg.sd, color="#1F5FBF")
        ax.set_xlabel("Drop in PR-AUC when the group is shuffled (held-out, 15 repeats)")
        plt.tight_layout()
        plt.savefig(OUT / "importance_groups.png", dpi=160)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
