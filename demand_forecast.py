"""Portfolio project 3: demand forecasting (time series ML).
Synthetic 2-year daily demand series; lag-feature regression with
Gradient Boosting; train/test evaluation; forecast charts. Seeded."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path(__file__).parent
rng = np.random.default_rng(11)

# ---------------- synthetic demand ----------------
days = pd.date_range("2024-10-01", "2026-09-30", freq="D")
t = np.arange(len(days))
dow = days.dayofweek.to_numpy()
doy = days.dayofyear.to_numpy()
demand = (200 + 0.08 * t                      # growth trend
          + 30 * np.sin(2 * np.pi * dow / 7)  # weekly seasonality
          + 45 * np.sin(2 * np.pi * doy / 365.25)  # yearly seasonality
          + rng.normal(0, 18, len(days)))
# promo spikes
for s in rng.choice(len(days), size=12, replace=False):
    demand[s:s + 5] += rng.normal(90, 15, 5)
demand = demand.clip(20)
df = pd.DataFrame({"date": days, "demand": demand})

# ---------------- features ----------------
for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = df["demand"].shift(lag)
df["roll_7"] = df["demand"].shift(1).rolling(7).mean()
df["roll_28"] = df["demand"].shift(1).rolling(28).mean()
df["dow"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df = df.dropna().reset_index(drop=True)

FEATS = [c for c in df.columns if c not in ("date", "demand")]
split = int(len(df) * 0.82)
tr, te = df.iloc[:split], df.iloc[split:]

model = GradientBoostingRegressor(random_state=11)
model.fit(tr[FEATS], tr["demand"])
pred = model.predict(te[FEATS])

mae = mean_absolute_error(te["demand"], pred)
rmse = mean_squared_error(te["demand"], pred) ** 0.5
mape = np.mean(np.abs((te["demand"] - pred) / te["demand"])) * 100
print(f"MAE={mae:.1f} RMSE={rmse:.1f} MAPE={mape:.1f}%  (test: {len(te)} days)")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. full history + forecast
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(tr["date"], tr["demand"], alpha=0.55, label="History (train)")
ax.plot(te["date"], te["demand"], color="black", linewidth=1.6, label="Actual (test)")
ax.plot(te["date"], pred, color="#d35400", linewidth=1.8, label="Forecast")
ax.set_title("Demand Forecast vs Actual — 130-Day Holdout", fontweight="bold", pad=12)
ax.set_ylabel("Daily demand (units)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "01_forecast_vs_actual.png", dpi=130)
plt.close(fig)

# 2. zoom on last 60 test days with error band
z = te.iloc[-60:]
zp = pred[-60:]
resid = np.abs(te["demand"].values - pred)
band = np.std(resid)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(z["date"], z["demand"], color="black", linewidth=1.8, label="Actual")
ax.plot(z["date"], zp, color="#d35400", linewidth=1.8, label="Forecast")
ax.fill_between(z["date"], zp - band, zp + band, color="#d35400", alpha=0.18, label="±1 std band")
ax.set_title("Forecast Detail — Last 60 Days", fontweight="bold", pad=12)
ax.set_ylabel("Daily demand (units)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "02_forecast_detail.png", dpi=130)
plt.close(fig)

# 3. feature importance
imp = pd.Series(model.feature_importances_, index=FEATS).sort_values().tail(8)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(imp.index, imp.values, color="#6c3483")
ax.set_title("What Drives the Forecast — Feature Importance", fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUT / "03_feature_importance.png", dpi=130)
plt.close(fig)

print("charts saved ->", OUT)
