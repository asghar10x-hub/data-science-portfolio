"""Sample portfolio project 1: E-commerce sales exploratory analysis.
Generates a synthetic 12-month order dataset, computes KPIs, and saves
publication-style charts. All data is synthetic (seeded)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
rng = np.random.default_rng(42)

# ---------------- synthetic data ----------------
n = 2000
categories = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books"]
products = {
    "Electronics": ["Wireless Headphones", "Smart Watch", "Bluetooth Speaker", "USB-C Hub"],
    "Clothing": ["Denim Jacket", "Running Shoes", "Cotton T-Shirt", "Winter Coat"],
    "Home & Kitchen": ["Air Fryer", "Ceramic Cookware Set", "Robot Vacuum", "Espresso Maker"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Camping Tent", "Road Bike"],
    "Books": ["Bestseller Novel", "Data Science Handbook", "Cookbook", "History Biography"],
}
segments = ["New", "Returning", "VIP"]
regions = ["North", "South", "East", "West"]

dates = pd.to_datetime(rng.choice(
    pd.date_range("2025-10-01", "2026-09-30", freq="D").astype(str), size=n))
cat = rng.choice(categories, size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
prod = [rng.choice(products[c]) for c in cat]
units = rng.integers(1, 6, size=n)
base_price = {"Electronics": 120, "Clothing": 45, "Home & Kitchen": 80, "Sports": 60, "Books": 25}
unit_price = np.array([rng.normal(base_price[c], base_price[c] * 0.25) for c in cat]).clip(5)
seg = rng.choice(segments, size=n, p=[0.45, 0.4, 0.15])
reg = rng.choice(regions, size=n)

df = pd.DataFrame({
    "order_id": np.arange(1, n + 1),
    "date": dates,
    "category": cat,
    "product": prod,
    "units": units,
    "unit_price": unit_price.round(2),
    "segment": seg,
    "region": reg,
})
df["revenue"] = (df["units"] * df["unit_price"]).round(2)
df["month"] = df["date"].dt.to_period("M").astype(str)

# ---------------- KPIs ----------------
total_rev = df["revenue"].sum()
aov = df["revenue"].mean()
print(f"Orders: {len(df):,} | Total revenue: ${total_rev:,.0f} | AOV: ${aov:,.2f}")
print("Top category:", df.groupby("category")["revenue"].sum().idxmax())

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. monthly revenue trend
m = df.groupby("month")["revenue"].sum()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(m.index, m.values, marker="o", linewidth=2.2)
ax.set_title("Monthly Revenue Trend (12 months)", fontweight="bold", pad=12)
ax.set_ylabel("Revenue (USD)")
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "01_monthly_revenue.png", dpi=130)
plt.close(fig)

# 2. revenue by category
c = df.groupby("category")["revenue"].sum().sort_values()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(c.index, c.values, color="#2f7de1")
ax.set_title("Revenue by Category", fontweight="bold", pad=12)
ax.set_xlabel("Revenue (USD)")
for i, v in enumerate(c.values):
    ax.text(v, i, f" ${v:,.0f}", va="center", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "02_revenue_by_category.png", dpi=130)
plt.close(fig)

# 3. top products
p = df.groupby("product")["revenue"].sum().sort_values().tail(8)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(p.index, p.values, color="#3aa655")
ax.set_title("Top 8 Products by Revenue", fontweight="bold", pad=12)
ax.set_xlabel("Revenue (USD)")
fig.tight_layout()
fig.savefig(OUT / "03_top_products.png", dpi=130)
plt.close(fig)

# 4. segment share donut
s = df.groupby("segment")["revenue"].sum()
fig, ax = plt.subplots(figsize=(6, 6))
wedges, _, autotext = ax.pie(
    s.values, labels=s.index, autopct="%1.0f%%", startangle=90,
    wedgeprops=dict(width=0.45, edgecolor="white"), pctdistance=0.8)
ax.set_title("Revenue Share by Customer Segment", fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUT / "04_segment_share.png", dpi=130)
plt.close(fig)

print("charts saved ->", OUT)
