"""Portfolio project 5: anomaly detection on business metrics.
Synthetic 365-day metric (e.g. daily revenue) with injected spikes/drops;
Isolation Forest detection; evaluation against known ground truth. Seeded."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

OUT = Path(__file__).parent
rng = np.random.default_rng(31)

# ---------------- synthetic metric ----------------
days = pd.date_range("2025-10-10", "2026-10-09", freq="D")
t = np.arange(len(days))
dow = days.dayofweek.to_numpy()
doy = days.dayofyear.to_numpy()
base = (5000 + 8 * t + 900 * np.sin(2 * np.pi * dow / 7)
        + 1200 * np.sin(2 * np.pi * doy / 365.25)
        + rng.normal(0, 350, len(days)))
truth = np.zeros(len(days), dtype=int)
for s in rng.choice(len(days), size=14, replace=False):  # spikes
    base[s:s + 2] += rng.normal(4500, 800, 2); truth[s:s + 2] = 1
for s in rng.choice(len(days), size=10, replace=False):  # drops
    base[s:s + 3] -= rng.normal(3800, 700, 3); truth[s:s + 3] = 1
df = pd.DataFrame({"date": days, "metric": base.clip(500), "is_anomaly": truth})
print(f"injected anomalies: {truth.sum()} days ({truth.mean():.1%})")

# ---------------- detection ----------------
df["dow"] = df["date"].dt.dayofweek
df["roll_7"] = df["metric"].shift(1).rolling(7).mean().bfill()
df["roll_28"] = df["metric"].shift(1).rolling(28).mean().bfill()
df["dev_7"] = (df["metric"] - df["roll_7"]) / df["roll_7"]
FEATS = ["metric", "dow", "roll_7", "roll_28", "dev_7"]

iso = IsolationForest(contamination=0.06, random_state=31)
df["score"] = -iso.fit(df[FEATS]).score_samples(df[FEATS])
df["flagged"] = (iso.fit_predict(df[FEATS]) == -1).astype(int)

p = precision_score(df["is_anomaly"], df["flagged"], zero_division=0)
r = recall_score(df["is_anomaly"], df["flagged"])
f1 = f1_score(df["is_anomaly"], df["flagged"])
print(f"precision={p:.2f} recall={r:.2f} f1={f1:.2f} "
      f"({df['flagged'].sum()} flagged vs {df['is_anomaly'].sum()} true)")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. metric with anomalies
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(df["date"], df["metric"], color="#2f7de1", linewidth=1.2, label="Daily metric")
ax.scatter(df.loc[df["flagged"] == 1, "date"], df.loc[df["flagged"] == 1, "metric"],
           color="#e74c3c", s=26, zorder=5, label="Detected anomaly")
ax.set_title("Anomaly Detection on Business Metric — Isolation Forest",
             fontweight="bold", pad=12)
ax.set_ylabel("Metric value")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "01_anomalies_timeline.png", dpi=130)
plt.close(fig)

# 2. score distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df.loc[df["is_anomaly"] == 0, "score"], bins=40, alpha=0.7,
        label="Normal days", color="#2f7de1")
ax.hist(df.loc[df["is_anomaly"] == 1, "score"], bins=20, alpha=0.85,
        label="True anomalies", color="#e74c3c")
ax.set_title("Anomaly Score Distribution", fontweight="bold", pad=12)
ax.set_xlabel("Anomaly score (higher = stranger)")
ax.set_ylabel("Days")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "02_score_distribution.png", dpi=130)
plt.close(fig)

# 3. precision/recall summary
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar(["Precision", "Recall", "F1"], [p, r, f1], color=["#2f7de1", "#3aa655", "#6c3483"])
ax.set_ylim(0, 1)
ax.set_title("Detection Quality vs Injected Ground Truth", fontweight="bold", pad=12)
for i, v in enumerate([p, r, f1]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "03_detection_quality.png", dpi=130)
plt.close(fig)

print("charts saved ->", OUT)
