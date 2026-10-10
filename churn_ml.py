"""Sample portfolio project 2: customer churn prediction (ML).
Builds a synthetic telecom-style dataset, trains Logistic Regression and
Random Forest, and saves evaluation charts. All data is synthetic (seeded)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)

OUT = Path(__file__).parent
rng = np.random.default_rng(7)

# ---------------- synthetic data ----------------
n = 3000
tenure = rng.integers(1, 73, size=n)
monthly = rng.normal(65, 25, size=n).clip(20, 150).round(2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.3, 0.2])
internet = rng.choice(["Fiber optic", "DSL", "No"], size=n, p=[0.45, 0.35, 0.2])
senior = rng.choice([0, 1], size=n, p=[0.85, 0.15])

# churn probability driven by intuitive factors
logit = (-1.2
         + 0.9 * (contract == "Month-to-month")
         + 0.5 * (internet == "Fiber optic")
         + 0.015 * monthly
         - 0.03 * tenure
         + 0.4 * senior)
prob = 1 / (1 + np.exp(-logit))
churn = (rng.random(n) < prob).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure, "monthly_charges": monthly,
    "contract": contract, "internet_service": internet,
    "senior_citizen": senior, "churn": churn,
})
print(f"rows={len(df):,} churn_rate={df['churn'].mean():.1%}")

X = df.drop(columns="churn")
y = df["churn"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

num = ["tenure_months", "monthly_charges", "senior_citizen"]
cat = ["contract", "internet_service"]
pre = ColumnTransformer([
    ("num", "passthrough", num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=7),
}
results = {}
probas = {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1]
    probas[name] = proba
    results[name] = {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
    }
    print(name, {k: f"{v:.3f}" for k, v in results[name].items()})

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. metrics comparison
labels = list(results["Logistic Regression"].keys())
x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w / 2, [results["Logistic Regression"][k] for k in labels], w, label="Logistic Regression")
ax.bar(x + w / 2, [results["Random Forest"][k] for k in labels], w, label="Random Forest")
ax.set_xticks(x, [l.capitalize() for l in labels])
ax.set_ylim(0, 1)
ax.set_title("Model Comparison — Churn Prediction", fontweight="bold", pad=12)
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "01_model_metrics.png", dpi=130)
plt.close(fig)

# 2. ROC curves
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for name, p_ in probas.items():
    fpr, tpr, _ = roc_curve(yte, p_)
    ax.plot(fpr, tpr, linewidth=2.2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("ROC Curves", fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "02_roc_curves.png", dpi=130)
plt.close(fig)

# 3. confusion matrix (random forest)
cm = confusion_matrix(yte, Pipeline([("pre", pre), ("clf", models["Random Forest"])]).fit(Xtr, ytr).predict(Xte))
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Predicted: Stay", "Predicted: Churn"])
ax.set_yticks([0, 1], ["Actual: Stay", "Actual: Churn"])
ax.set_title("Confusion Matrix — Random Forest", fontweight="bold", pad=12)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUT / "03_confusion_matrix.png", dpi=130)
plt.close(fig)

# 4. feature importance (random forest)
rf = Pipeline([("pre", pre), ("clf", models["Random Forest"])]).fit(Xtr, ytr)
feat_names = (num + list(rf.named_steps["pre"].named_transformers_["cat"]
                         .get_feature_names_out(cat)))
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat_names).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp.index.str.replace("_", " "), imp.values, color="#c26a1b")
ax.set_title("Feature Importance — Random Forest", fontweight="bold", pad=12)
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(OUT / "04_feature_importance.png", dpi=130)
plt.close(fig)

print("charts saved ->", OUT)
