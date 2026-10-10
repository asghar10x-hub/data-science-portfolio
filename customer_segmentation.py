"""Portfolio project 4: customer segmentation (RFM + K-Means).
Synthetic customer base; RFM features; K-Means clustering; PCA visualization;
business interpretation of segments. Seeded."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

OUT = Path(__file__).parent
rng = np.random.default_rng(21)

# ---------------- synthetic customers (4 latent groups) ----------------
n = 800
group = rng.choice(4, size=n, p=[0.3, 0.25, 0.25, 0.2])
recency = np.where(group == 0, rng.integers(1, 30, n),
          np.where(group == 1, rng.integers(60, 200, n),
          np.where(group == 2, rng.integers(1, 45, n), rng.integers(200, 400, n))))
frequency = np.where(group == 0, rng.integers(8, 25, n),
            np.where(group == 1, rng.integers(5, 15, n),
            np.where(group == 2, rng.integers(1, 4, n), rng.integers(1, 6, n))))
monetary = np.where(group == 0, rng.normal(2500, 600, n),
           np.where(group == 1, rng.normal(1200, 400, n),
           np.where(group == 2, rng.normal(300, 120, n), rng.normal(500, 200, n)))).clip(20)

df = pd.DataFrame({"recency_days": recency, "frequency": frequency,
                   "monetary": monetary.round(2)})

X = StandardScaler().fit_transform(df[["recency_days", "frequency", "monetary"]])
km = KMeans(n_clusters=4, n_init=20, random_state=21)
df["cluster"] = km.fit_predict(X)
print("inertia:", round(km.inertia_, 1))
print(df.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean().round(1))
print("sizes:", df["cluster"].value_counts().sort_index().tolist())

# business labels by profile order
profiles = df.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()
order = profiles.sort_values(["monetary", "frequency"], ascending=False).index.tolist()
names = {order[0]: "Champions", order[1]: "Loyal Regulars",
         order[2]: "At-Risk", order[3]: "Dormant"}
df["segment"] = df["cluster"].map(names)
for s, g in df.groupby("segment"):
    print(f"{s}: n={len(g)}, R={g['recency_days'].mean():.0f}d "
          f"F={g['frequency'].mean():.1f} M=${g['monetary'].mean():.0f}")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
colors = {"Champions": "#2f7de1", "Loyal Regulars": "#3aa655",
          "At-Risk": "#e67e22", "Dormant": "#8e8e8e"}

# 1. PCA scatter
xy = PCA(n_components=2, random_state=21).fit_transform(X)
fig, ax = plt.subplots(figsize=(8, 6))
for s, c in colors.items():
    m = df["segment"] == s
    ax.scatter(xy[m, 0], xy[m, 1], c=c, label=s, alpha=0.65, s=28)
ax.set_title("Customer Segments — PCA Projection", fontweight="bold", pad=12)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "01_segment_scatter.png", dpi=130)
plt.close(fig)

# 2. segment profiles
prof = df.groupby("segment")[["recency_days", "frequency", "monetary"]].mean()
prof = prof.reindex(colors.keys())
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=False)
for ax, col, title in zip(axes, prof.columns,
                          ["Avg Recency (days ↓ better)", "Avg Frequency",
                           "Avg Monetary ($)"]):
    ax.bar(prof.index, prof[col], color=[colors[s] for s in prof.index])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=18)
fig.suptitle("Segment Profiles — RFM Means", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "02_segment_profiles.png", dpi=130)
plt.close(fig)

# 3. segment sizes
sizes = df["segment"].value_counts().reindex(colors.keys())
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(sizes.index, sizes.values, color=[colors[s] for s in sizes.index])
ax.set_title("Customers per Segment", fontweight="bold", pad=12)
ax.set_ylabel("Customers")
ax.tick_params(axis="x", rotation=15)
for i, v in enumerate(sizes.values):
    ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "03_segment_sizes.png", dpi=130)
plt.close(fig)

print("charts saved ->", OUT)
