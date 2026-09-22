"""Regenerate just the case distribution figure with legend outside the plot."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.3, rc={
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

CASE_COLORS = {
    "A": "#E63946", "B": "#2A9D8F", "C": "#264653",
    "D": "#E9C46A", "None": "#AAAAAA",
}
CASE_LABELS = {
    "A": "A (Renter pressure)", "B": "B (Speculative)",
    "C": "C (Active gentrification)", "D": "D (Degradation)",
    "None": "None (No change)",
}

print("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year
grouped = df.groupby("CUSEC")
df["rent_change"] = grouped["unitprice_residential_rent_all"].pct_change()
df["sale_change"] = grouped["unitprice_residential_sale_all"].pct_change()
df = classify_cases(df)

clean = df[df["case"].isin(["A", "B", "C", "D", "None"])].copy()
case_time = clean.groupby(["year", "case"]).size().unstack(fill_value=0)
case_order_stack = ["D", "A", "None", "B", "C"]
case_time = case_time[[c for c in case_order_stack if c in case_time.columns]]

fig, ax = plt.subplots(figsize=(14, 7))
bottom = np.zeros(len(case_time))
for case in case_time.columns:
    vals = case_time[case].values
    ax.bar(case_time.index.astype(str), vals, bottom=bottom,
           color=CASE_COLORS[case], label=CASE_LABELS.get(case, case),
           edgecolor="white", linewidth=0.5)
    bottom += vals
ax.set_xlabel("Year", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Census Sections", fontsize=14, fontweight="bold")
ax.set_title("Gentrification Case Distribution Over Time", fontsize=16, fontweight="bold", pad=15)
ax.legend(title="Case Type", title_fontsize=12, fontsize=11,
          loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.95, edgecolor="#cccccc")
ax.tick_params(axis="both", labelsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
sns.despine(left=True, bottom=True)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "case_distribution_over_time.png", bbox_inches="tight")
plt.close(fig)
print("Saved case_distribution_over_time.png")
