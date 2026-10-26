"""Subgroup fairness audit for Head A (retention model) on the held-out slice (spec section 3.7).

Reuses the chronological split of scripts/retrain.py and the deployed artifact in models/.
Positive class = RETAINED (the model's target). A customer is predicted retained when P(retained) >= 0.82;
below 0.82 the customer is routed to staff review for a retention offer (offer rate = 1 - selection rate).

Per subgroup (region, tenure band, spend quartile): n, recall (of truly retained customers), precision,
selection rate, plus 95% bootstrap CI on recall. Any group whose recall is more than 5 percentage points
below the overall recall is flagged. Then four mitigations are evaluated on the same held-out slice:
  A  group-specific thresholds (chosen on the calibration slice to equalise the selection rate)
  B  rank within group: review the lowest-scoring customers of each group up to a fixed review budget
  C  per-group isotonic recalibration (fitted on the calibration slice) with the global 0.82 threshold
  D  retrain with inverse-group-frequency sample weights (same recipe otherwise)
Nothing here changes the deployed model.   Run: python scripts/fairness_audit.py  -> reports/fairness/
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from retrain import DATA, CURRENT_ART, ID, PARAMS, THRESHOLD, prepare, split  # noqa: E402

OUT = Path("reports/fairness")
RNG = np.random.default_rng(42)
MIN_N, FLAG_PP = 30, 0.05
EUROPE = {"Austria", "Belgium", "Channel Islands", "Cyprus", "Czech Republic", "Denmark", "EIRE", "European Community",
          "Finland", "France", "Germany", "Greece", "Iceland", "Italy", "Lithuania", "Malta", "Netherlands", "Norway",
          "Poland", "Portugal", "Spain", "Sweden", "Switzerland"}


def segments(X: pd.DataFrame, ref: pd.DataFrame) -> dict[str, pd.Series]:
    q = ref["monetary_total"].quantile([.25, .5, .75]).values  # quartile cut-offs from the TRAINING slice only
    spend = pd.cut(X["monetary_total"], [-np.inf, *q, np.inf], labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype(str)
    tenure = pd.cut(X["customer_lifetime_days"], [-1, 90, 365, 10**6], labels=["0-90 days", "91-365 days", "366+ days"]).astype(str)
    region = X["country"].map(lambda c: "UK" if c == "United Kingdom" else ("Europe (non-UK)" if c in EUROPE else "Rest of world"))
    return {"spend_quartile": spend, "tenure_band": tenure, "region": region}


def metrics(y: pd.Series, sel: np.ndarray) -> dict:
    y = y.to_numpy()
    sel = np.asarray(sel, bool)
    tp = int((sel & (y == 1)).sum())
    return dict(n=int(len(y)), retained_rate=float((y == 1).mean()), selection_rate=float(sel.mean()),
                recall=tp / max(int((y == 1).sum()), 1) if (y == 1).any() else np.nan,
                precision=tp / max(int(sel.sum()), 1) if sel.any() else np.nan)


def recall_ci(y: pd.Series, sel: np.ndarray, B: int = 1000):
    y = y.to_numpy()
    sel = np.asarray(sel, bool)
    v = []
    for _ in range(B):
        i = RNG.integers(0, len(y), len(y))
        pos = y[i] == 1
        if pos.any():
            v.append(sel[i][pos].mean())
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def audit_table(y, sel, segs, with_ci=True) -> pd.DataFrame:
    overall = metrics(y, sel)
    rows = [dict(group_type="ALL", group="all", **overall)]
    for gt, s in segs.items():
        for gv in sorted(s.unique()):
            m = (s == gv).to_numpy()
            r = metrics(y[m], sel[m])
            lo, hi = recall_ci(y[m], sel[m]) if with_ci else (np.nan, np.nan)
            rows.append(dict(group_type=gt, group=gv, **r, recall_lo=lo, recall_hi=hi,
                             recall_gap_pp=100 * (r["recall"] - overall["recall"]) if r["recall"] == r["recall"] else np.nan,
                             flagged=bool(r["recall"] == r["recall"] and overall["recall"] - r["recall"] > FLAG_PP),
                             small=bool(r["n"] < MIN_N)))
    return pd.DataFrame(rows)


def summarise(name, y, sel, segs, extra=None) -> dict:
    t = audit_table(y, sel, segs, with_ci=False)
    allr = t.iloc[0]
    g = t.iloc[1:]
    g = g[~g.small.astype(bool)]
    return dict(option=name, selection_rate=allr.selection_rate, recall=allr.recall, precision=allr.precision,
                worst_group_recall=float(g.recall.min()), groups_flagged=int(g.flagged.astype(bool).sum()), **(extra or {}))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = prepare(df)
    (Xtr, ytr), (Xca, yca), (Xte, yte) = split(X, y, df[ID])
    model = joblib.load(CURRENT_ART)
    pte = model.predict_proba(Xte)[:, 1]
    pca = model.predict_proba(Xca)[:, 1]
    seg_te, seg_ca = segments(Xte, Xtr), segments(Xca, Xtr)

    base_sel = pte >= THRESHOLD
    res = audit_table(yte, base_sel, seg_te)
    res.to_csv(OUT / "subgroup_metrics.csv", index=False)
    rows = [summarise("Baseline (single threshold 0.82)", yte, base_sel, seg_te,
                      dict(pr_auc=float(average_precision_score(yte, pte)), roc_auc=float(roc_auc_score(yte, pte))))]

    # A: group-specific thresholds on the spend-quartile segmentation (selection rate matched to the overall
    # calibration selection rate)
    gt = "spend_quartile"
    target = float((pca >= THRESHOLD).mean())
    thr = {}
    for gv in sorted(seg_ca[gt].unique()):
        thr[gv] = float(np.quantile(pca[(seg_ca[gt] == gv).to_numpy()], 1 - target))
    selA = np.array([p >= thr[g] for p, g in zip(pte, seg_te[gt], strict=True)])
    rows.append(summarise("A. Group thresholds (spend quartile)", yte, selA, seg_te, dict(pr_auc=np.nan, roc_auc=np.nan)))
    pd.DataFrame([dict(group=k, threshold=v) for k, v in thr.items()]).to_csv(OUT / "mitigation_A_thresholds.csv", index=False)

    # B: fixed review budget, spread across groups in proportion to size, lowest scores first
    budget = float((pte < THRESHOLD).mean())
    selB = np.ones(len(pte), bool)
    for gv in seg_te[gt].unique():
        idx = np.where((seg_te[gt] == gv).to_numpy())[0]
        k = int(round(budget * len(idx)))
        selB[idx[np.argsort(pte[idx])[:k]]] = False
    rows.append(summarise("B. Rank within group, same review budget", yte, selB, seg_te, dict(pr_auc=np.nan, roc_auc=np.nan)))

    # C: per-group isotonic recalibration
    cal = {gv: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pca[(seg_ca[gt] == gv).to_numpy()], yca[(seg_ca[gt] == gv).to_numpy()].to_numpy())
           for gv in seg_ca[gt].unique() if (seg_ca[gt] == gv).sum() >= MIN_N}
    pC = np.array([cal[g].predict([p])[0] if g in cal else p for p, g in zip(pte, seg_te[gt], strict=True)])
    rows.append(summarise("C. Per-group recalibration, threshold 0.82", yte, pC >= THRESHOLD, seg_te,
                          dict(pr_auc=float(average_precision_score(yte, pC)), roc_auc=float(roc_auc_score(yte, pC)))))

    # D: retrain with inverse-group-frequency weights (spend quartile)
    sq = segments(Xtr, Xtr)[gt]
    w = 1.0 / sq.map(sq.value_counts(normalize=True)).to_numpy()
    w = w / w.mean()
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.frozen import FrozenEstimator
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numeric = [c for c in Xtr.columns if c != "country"]
    pre = ColumnTransformer([("numeric", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), numeric),
                             ("categorical", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), ["country"])])
    pipe = Pipeline([("preprocess", pre), ("model", ExtraTreesClassifier(**PARAMS))]).fit(Xtr, ytr, model__sample_weight=w)
    calD = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(Xca, yca)
    pD = calD.predict_proba(Xte)[:, 1]
    rows.append(summarise("D. Retrain with group weights", yte, pD >= THRESHOLD, seg_te,
                          dict(pr_auc=float(average_precision_score(yte, pD)), roc_auc=float(roc_auc_score(yte, pD)))))
    mit = pd.DataFrame(rows)
    mit.to_csv(OUT / "mitigation_options.csv", index=False)
    # per-group view of every option (for the report)
    detail = []
    for name, sel in [("Baseline", base_sel), ("A", selA), ("B", selB), ("C", pC >= THRESHOLD), ("D", pD >= THRESHOLD)]:
        t = audit_table(yte, sel, seg_te, with_ci=False)
        t.insert(0, "option", name)
        detail.append(t)
    pd.concat(detail).to_csv(OUT / "mitigation_by_group.csv", index=False)

    (OUT / "summary.json").write_text(json.dumps(dict(threshold=THRESHOLD, held_out_n=int(len(yte)),
                                                      overall=res.iloc[0].to_dict(), flag_rule_pp=FLAG_PP * 100), indent=2, default=float))
    ablation_country(Xtr, ytr, Xca, yca, Xte, yte, pte)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sub = res[res.group_type != "ALL"].copy()
        sub["label"] = sub.group_type.str.replace("_", " ") + ": " + sub.group
        fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for a, col, t in [(ax[0], "recall", "Recall of retained customers"), (ax[1], "selection_rate", "Selection rate (predicted retained)")]:
            cols = ["#C0392B" if f else "#1F5FBF" for f in sub.flagged]
            a.barh(sub.label, sub[col], color=cols if col == "recall" else "#1F5FBF")
            a.set_title(t, fontsize=10)
            a.set_xlim(0, 1)
            a.axvline(res.iloc[0][col], color="black", ls="--", lw=1)
        plt.tight_layout()
        plt.savefig(OUT / "fairness_subgroups.png", dpi=160)
    except Exception as e:
        print("plot skipped:", e)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())
    print(mit.round(3).to_string())


def ablation_country(Xtr, ytr, Xca, yca, Xte, yte, pte) -> None:
    """Data-minimisation check: does the model need country / is_uk?"""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.frozen import FrozenEstimator
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import brier_score_loss
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in Xtr.columns if c not in ("country", "is_uk")]
    pre = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), cols)])
    pipe = Pipeline([("p", pre), ("m", ExtraTreesClassifier(**PARAMS))]).fit(Xtr[cols], ytr)
    p = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(Xca[cols], yca).predict_proba(Xte[cols])[:, 1]

    def f(q):
        return dict(pr_auc=float(average_precision_score(yte, q)), roc_auc=float(roc_auc_score(yte, q)),
                    brier=float(brier_score_loss(yte, q)), selection_rate=float((q >= THRESHOLD).mean()),
                    recall=float((q >= THRESHOLD)[yte.to_numpy() == 1].mean()))

    (OUT / "ablation_country.json").write_text(json.dumps(dict(with_country=f(pte), without_country=f(p)), indent=2))


if __name__ == "__main__":
    main()
