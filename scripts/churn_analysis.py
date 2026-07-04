"""
Customer Churn & Revenue Recovery — Synthetic Analysis Pipeline
================================================================
This script reconstructs the methodology of a real B2B/OEM churn-recovery
project on ***synthetic data*** that mirrors the structure of ERPNext Sales
Invoices (customer, invoice date, amount, territory). NO real company data is
used. Figures are tuned to resemble the real engagement for illustration.

Pipeline:
  1. Generate synthetic Sales Invoice data (~16k customers, 5 territories)
  2. Classify status: Active (<90d) / Dormant (90-180d) / Churned (>180d)
  3. ABC segmentation (A = top ~70% of revenue, Pareto)
  4. RFM scoring & segment labels (Champions, Loyal, At-Risk, Lost, ...)
  5. Simulate re-engagement recovery (~2k accounts)
  6. Render an executive dashboard PNG + supporting charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
AS_OF = datetime(2025, 6, 30)                 # analysis reference date (mid-2025)
START = datetime(2023, 7, 1)                   # 2 years of history
TERRITORIES = ["Coimbatore", "Chennai", "Hyderabad", "Mumbai", "Ahmedabad"]
N_CUST = 16200
TARGET_TURNOVER_CR = 130                        # ₹130 Cr total turnover

# Palette (matches portfolio blue→purple→teal)
C_PRIMARY = "#4361ee"; C_TEAL = "#4cc9f0"; C_PURPLE = "#7209b7"
C_AMBER = "#f8961e"; C_RED = "#e63946"; C_GREEN = "#2a9d8f"
INK = "#1d2433"; MUTED = "#6b7280"; CARD = "#ffffff"; BG = "#f4f6fb"

# ---------------------------------------------------------------------------
# 1. Generate synthetic Sales Invoices
# ---------------------------------------------------------------------------
terr_weights = [0.30, 0.22, 0.18, 0.17, 0.13]
cust_terr = np.random.choice(TERRITORIES, N_CUST, p=terr_weights)

# Each customer has a "days since last purchase" drawn to hit target churn mix
# ~44% churned (>180d) -> ~7.1k ; rest dormant/active
u = np.random.rand(N_CUST)
recency = np.where(u < 0.30, np.random.randint(1, 90, N_CUST),        # active
          np.where(u < 0.56, np.random.randint(90, 181, N_CUST),      # dormant
                             np.random.randint(181, 730, N_CUST)))     # churned
last_purchase = np.array([AS_OF - timedelta(days=int(d)) for d in recency])

# Frequency & monetary — skewed (few big accounts drive most revenue = Pareto)
frequency = np.random.gamma(shape=1.6, scale=6, size=N_CUST).astype(int) + 1
monetary = np.round(np.random.lognormal(mean=11.2, sigma=1.15, size=N_CUST), 0)

cust = pd.DataFrame({
    "customer_id": [f"CUST-{i:05d}" for i in range(N_CUST)],
    "territory": cust_terr,
    "recency_days": recency,
    "last_invoice_date": last_purchase,
    "frequency": frequency,
    "monetary": monetary,
})

# Scale monetary so total turnover ≈ ₹130 Cr
scale = (TARGET_TURNOVER_CR * 1e7) / cust["monetary"].sum()
cust["monetary"] = (cust["monetary"] * scale).round(0)

# ---------------------------------------------------------------------------
# 2. Status classification
# ---------------------------------------------------------------------------
def status(r):
    if r < 90:  return "Active"
    if r < 180: return "Dormant"
    return "Churned"
cust["status"] = cust["recency_days"].apply(status)

# ---------------------------------------------------------------------------
# 3. ABC segmentation (Pareto on revenue)
# ---------------------------------------------------------------------------
cust = cust.sort_values("monetary", ascending=False).reset_index(drop=True)
cust["cum_rev_pct"] = cust["monetary"].cumsum() / cust["monetary"].sum()
cust["ABC"] = np.where(cust["cum_rev_pct"] <= 0.70, "A",
              np.where(cust["cum_rev_pct"] <= 0.90, "B", "C"))

# ---------------------------------------------------------------------------
# 4. RFM scoring (1-5 quintiles) & segment labels
# ---------------------------------------------------------------------------
cust["R"] = pd.qcut(cust["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
cust["F"] = pd.qcut(cust["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
cust["M"] = pd.qcut(cust["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

def rfm_segment(row):
    r, f, m = row.R, row.F, row.M
    if r >= 4 and f >= 4:            return "Champions"
    if r >= 3 and f >= 3:            return "Loyal"
    if r >= 4 and f <= 2:            return "New / Promising"
    if r <= 2 and f >= 3 and m >= 3: return "At-Risk (High Value)"
    if r <= 2 and m >= 3:            return "Can't Lose Them"
    if r <= 2 and f <= 2:            return "Lost / Hibernating"
    return "Needs Attention"
cust["rfm_segment"] = cust.apply(rfm_segment, axis=1)

# ---------------------------------------------------------------------------
# 5. Simulate re-engagement recovery (~2k churned accounts recovered)
# ---------------------------------------------------------------------------
churned = cust[cust["status"] == "Churned"].copy()
# Prioritise recovery by value (A/B accounts recovered more often)
recover_prob = churned["ABC"].map({"A": 0.55, "B": 0.30, "C": 0.14}).values
recovered_mask = np.random.rand(len(churned)) < recover_prob
recovered_ids = churned.loc[recovered_mask, "customer_id"]
# cap ~2050
recovered_ids = recovered_ids.iloc[:2050]
cust["recovered"] = cust["customer_id"].isin(recovered_ids)

total_turnover = cust["monetary"].sum()
# Recovered REVENUE = incremental re-engagement contribution to turnover (~8%),
# NOT the lifetime value of recovered accounts. Matches the real ~8% metric.
recovery_share = 0.08
recovered_revenue = recovery_share * total_turnover
n_churned = int((cust.status == "Churned").sum())
n_recovered = int(cust.recovered.sum())
recovery_rate = n_recovered / n_churned

# ---------------------------------------------------------------------------
# 6. Export a small synthetic sample CSV (invoice-level, 2k rows)
# ---------------------------------------------------------------------------
sample = cust.sample(2000, random_state=1).copy()
inv_rows = []
for _, c in sample.iterrows():
    n = min(int(c.frequency), 6)
    for j in range(max(n, 1)):
        d = c.last_invoice_date - timedelta(days=int(np.random.randint(0, 500)))
        if d < START: d = START + timedelta(days=int(np.random.randint(0, 60)))
        inv_rows.append({
            "invoice_id": f"SINV-{np.random.randint(10000,99999)}",
            "customer_id": c.customer_id,
            "territory": c.territory,
            "invoice_date": d.strftime("%Y-%m-%d"),
            "amount_inr": int(max(c.monetary / max(n,1) * np.random.uniform(0.6,1.4), 500)),
        })
pd.DataFrame(inv_rows).to_csv("data/sample_sales_invoices.csv", index=False)

# ---------------------------------------------------------------------------
# 7. Print summary (used in README)
# ---------------------------------------------------------------------------
print("=== SUMMARY (synthetic) ===")
print("Total customers      :", len(cust))
print("Churned (>180d)      :", (cust.status=='Churned').sum())
print("Dormant (90-180d)    :", (cust.status=='Dormant').sum())
print("Active (<90d)        :", (cust.status=='Active').sum())
print("Recovered            :", cust.recovered.sum())
print("Total turnover (Cr)  :", round(total_turnover/1e7,1))
print("Recovered rev (Cr)   :", round(recovered_revenue/1e7,1))
print("Recovery share       :", f"{recovery_share*100:.1f}%")
print("A accounts share rev :", f"{cust[cust.ABC=='A'].monetary.sum()/total_turnover*100:.0f}%",
      "| A count:", (cust.ABC=='A').sum())

# ===========================================================================
# 8. DASHBOARD RENDER
# ===========================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#e5e7eb", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
})

fig = plt.figure(figsize=(16, 10.2), dpi=130)
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(4, 6, figure=fig, height_ratios=[0.6, 0.75, 1.25, 1.25],
                       hspace=0.55, wspace=0.55, left=0.035, right=0.965, top=0.955, bottom=0.06)

def card(ax, title, value, sub, accent):
    ax.axis("off")
    box = FancyBboxPatch((0.02, 0.08), 0.96, 0.84, boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=0, facecolor=CARD, mutation_aspect=0.5)
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.05, 0.84, boxstyle="round,pad=0,rounding_size=0.02",
                                linewidth=0, facecolor=accent))
    ax.text(0.12, 0.66, value, fontsize=21, fontweight="bold", va="center", color=INK)
    ax.text(0.12, 0.36, title, fontsize=10.5, va="center", color=INK, fontweight="600")
    ax.text(0.12, 0.19, sub, fontsize=8.8, va="center", color=MUTED)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

# --- Header ---
hax = fig.add_subplot(gs[0, :]); hax.axis("off")
hax.text(0.0, 0.62, "Customer Churn & Revenue Recovery — Executive Dashboard",
         fontsize=20, fontweight="bold", color=INK, va="center")
hax.text(0.0, 0.16, "B2B / OEM Manufacturing  ·  FY 2024–25  ·  5 Territories  ·  Source: ERPNext Sales Invoices "
         "(synthetic sample — no company data)", fontsize=10.5, color=MUTED, va="center")
hax.text(1.0, 0.5, "Abilash K S", fontsize=11, color=C_PRIMARY, fontweight="bold", ha="right", va="center")
hax.set_xlim(0,1); hax.set_ylim(0,1)

# --- KPI cards ---
a_share = cust[cust.ABC=='A'].monetary.sum()/total_turnover*100
kpis = [
    ("Total Customers", f"{len(cust):,}", "active book of accounts", C_PRIMARY),
    ("Churned (>180d)", f"{n_churned:,}", f"{n_churned/len(cust)*100:.0f}% of base at risk", C_RED),
    ("Recovered", f"{n_recovered:,}", "re-engaged accounts", C_GREEN),
    ("Revenue Recovered", "≈8%", f"of ₹130 Cr turnover (~₹{recovered_revenue/1e7:.0f} Cr)", C_PURPLE),
    ("Recovery Rate", f"{recovery_rate*100:.1f}%", "of churned reactivated", C_TEAL),
    ("A-Class Accounts", f"{a_share:.0f}%", "of revenue from top tier", C_AMBER),
]
for i, (t, v, s, a) in enumerate(kpis):
    card(fig.add_subplot(gs[1, i]), t, v, s, a)

# --- Chart 1: Customer status funnel ---
ax1 = fig.add_subplot(gs[2, 0:2])
stat = cust["status"].value_counts().reindex(["Active", "Dormant", "Churned"])
recov = cust["recovered"].sum()
labels = ["Active\n(<90d)", "Dormant\n(90–180d)", "Churned\n(>180d)", "Recovered\n(re-engaged)"]
vals = [stat["Active"], stat["Dormant"], stat["Churned"], recov]
cols = [C_GREEN, C_AMBER, C_RED, C_PRIMARY]
bars = ax1.barh(labels[::-1], vals[::-1], color=cols[::-1], height=0.62)
for b, v in zip(bars, vals[::-1]):
    ax1.text(v + 120, b.get_y()+b.get_height()/2, f"{v:,}", va="center", fontsize=10, color=INK, fontweight="600")
ax1.set_title("Customer Lifecycle Status", fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax1.set_xlim(0, max(vals)*1.18); ax1.spines[["top","right"]].set_visible(False)
ax1.set_facecolor(CARD); ax1.tick_params(length=0)

# --- Chart 2: ABC Pareto ---
ax2 = fig.add_subplot(gs[2, 2:4])
abc = cust.groupby("ABC")["monetary"].agg(["sum","count"]).reindex(["A","B","C"])
abc_share = (abc["sum"]/abc["sum"].sum()*100)
b2 = ax2.bar(["A","B","C"], abc_share, color=[C_PRIMARY,C_TEAL,"#cbd5e1"], width=0.6)
for bar, val, cnt in zip(b2, abc_share, abc["count"]):
    ax2.text(bar.get_x()+bar.get_width()/2, val+1.5, f"{val:.0f}%", ha="center", fontsize=10.5, fontweight="bold", color=INK)
    ax2.text(bar.get_x()+bar.get_width()/2, 4, f"{cnt:,} custs", ha="center", fontsize=8.6, color="white", fontweight="600")
ax2.set_title("ABC Revenue Concentration (Pareto)", fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax2.set_ylabel("% of revenue"); ax2.set_ylim(0,80)
ax2.spines[["top","right"]].set_visible(False); ax2.set_facecolor(CARD); ax2.tick_params(length=0)

# --- Chart 3: Churn by territory ---
ax3 = fig.add_subplot(gs[2, 4:6])
tt = cust.groupby("territory")["status"].apply(lambda s:(s=="Churned").mean()*100).reindex(TERRITORIES)
tc = cust.groupby("territory")["customer_id"].count().reindex(TERRITORIES)
b3 = ax3.bar(TERRITORIES, tt, color=C_PURPLE, width=0.6)
for bar, val in zip(b3, tt):
    ax3.text(bar.get_x()+bar.get_width()/2, val+0.8, f"{val:.0f}%", ha="center", fontsize=9.5, fontweight="600", color=INK)
ax3.set_title("Churn Rate by Territory", fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax3.set_ylabel("% churned"); ax3.set_ylim(0, tt.max()*1.25)
ax3.spines[["top","right"]].set_visible(False); ax3.set_facecolor(CARD)
ax3.tick_params(length=0); ax3.set_xticks(range(len(TERRITORIES)))
ax3.set_xticklabels(TERRITORIES, rotation=20, ha="right", fontsize=9)

# --- Chart 4: RFM segments ---
ax4 = fig.add_subplot(gs[3, 0:3])
seg = cust["rfm_segment"].value_counts()
order = ["Champions","Loyal","New / Promising","Needs Attention","At-Risk (High Value)","Can't Lose Them","Lost / Hibernating"]
seg = seg.reindex([s for s in order if s in seg.index])
seg_colors = {"Champions":C_GREEN,"Loyal":C_TEAL,"New / Promising":C_PRIMARY,"Needs Attention":C_AMBER,
              "At-Risk (High Value)":"#ef476f","Can't Lose Them":C_PURPLE,"Lost / Hibernating":"#94a3b8"}
b4 = ax4.barh(seg.index[::-1], seg.values[::-1], color=[seg_colors[s] for s in seg.index[::-1]], height=0.66)
for bar, v in zip(b4, seg.values[::-1]):
    ax4.text(v+80, bar.get_y()+bar.get_height()/2, f"{v:,}", va="center", fontsize=9.2, color=INK, fontweight="600")
ax4.set_title("RFM Behavioural Segments  (prioritisation for re-engagement)", fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax4.set_xlim(0, seg.max()*1.16); ax4.spines[["top","right"]].set_visible(False)
ax4.set_facecolor(CARD); ax4.tick_params(length=0)

# --- Chart 5: Recovered revenue trend over re-engagement window ---
ax5 = fig.add_subplot(gs[3, 3:6])
months = pd.date_range("2024-07-01","2025-06-01", freq="MS")
np.random.seed(7)
base = np.linspace(0.3, 1.0, len(months))
monthly = (base * (recovered_revenue/1e7/ base.sum())) * np.random.uniform(0.85,1.15,len(months))
cum = np.cumsum(monthly)
ax5.fill_between(months, cum, color=C_PRIMARY, alpha=0.12)
ax5.plot(months, cum, color=C_PRIMARY, lw=2.4, marker="o", ms=4)
ax5.text(months[-1], cum[-1], f"  ₹{cum[-1]:.1f} Cr", va="center", fontsize=10, color=C_PRIMARY, fontweight="bold")
ax5.set_title("Cumulative Recovered Revenue (re-engagement window)", fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax5.set_ylabel("₹ Crore"); ax5.spines[["top","right"]].set_visible(False)
ax5.set_facecolor(CARD); ax5.tick_params(length=0)
import matplotlib.dates as mdates
ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y")); ax5.tick_params(axis="x", labelsize=8, rotation=0)

plt.savefig("assets/dashboard_overview.png", facecolor=BG, bbox_inches="tight", dpi=130)
print("\nSaved assets/dashboard_overview.png")
plt.close()
