"""D3 — Generate reliability and profit-curve figures for the report."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("docs").mkdir(exist_ok=True)

# --- Reliability (calibration) curves ---
rel = pd.read_csv("docs/calibration_reliability.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, model in zip(axes, rel["model"].unique()):
    sub = rel[rel["model"] == model]
    for stage, marker in [("before_calibration", "o--"), ("after_calibration", "s-")]:
        s = sub[sub["stage"] == stage].sort_values("mean_predicted")
        ax.plot(s["mean_predicted"], s["fraction_positive"], marker,
                label=stage.replace("_", " "))
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="perfect calibration")
    ax.set_title(model)
    ax.set_xlabel("Mean predicted P(retain)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("Observed fraction retained")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Reliability curves on held-out test, before vs. after calibration")
fig.tight_layout()
fig.savefig("docs/head_a_reliability_curves.png", dpi=150)
print("Saved docs/head_a_reliability_curves.png")

# --- Profit threshold curve ---
profit = pd.read_csv("docs/profit_threshold_sweep.csv")
fig2, ax2 = plt.subplots(figsize=(7, 5))
for model, grp in profit.groupby("model"):
    ax2.plot(grp["threshold"], grp["expected_profit"], marker=".", label=model)
    best = grp.loc[grp["expected_profit"].idxmax()]
    ax2.scatter([best["threshold"]], [best["expected_profit"]], zorder=5)
    ax2.annotate(f"  optimum t={best['threshold']:.2f}\n  ${best['expected_profit']:,.0f}",
                 (best["threshold"], best["expected_profit"]), fontsize=8)
ax2.axhline(0, color="grey", linewidth=0.8)
ax2.set_xlabel("Decision threshold t (target if P(retain) < t)")
ax2.set_ylabel("Expected profit on held-out test (CAD)")
ax2.set_title("Expected-profit curve by targeting threshold")
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig("docs/head_a_profit_curve.png", dpi=150)
print("Saved docs/head_a_profit_curve.png")
