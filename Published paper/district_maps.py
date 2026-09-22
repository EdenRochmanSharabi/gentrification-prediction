"""
Generate district-level maps for Madrid and Barcelona showing
gentrification type distribution at the district level.
Uses district boundary geometries from city open data portals.
"""
import sys, os, warnings, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import geopandas as gpd
import urllib.request

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
CASE_ORDER = ["A", "B", "C", "D", "None"]

# ─── Load and classify data ───
print("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
grouped = df.groupby("CUSEC")
df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)

clean = df[df["case"].isin(CASE_ORDER)].copy()

# ─── Madrid city data ───
madrid = clean[(clean["CPRO"] == "28") & (clean["CMUN"] == "079")].copy()
print(f"Madrid city: {len(madrid):,} observations, {madrid['CUDIS'].nunique()} districts")

# ─── Barcelona city data ───
bcn = clean[(clean["CPRO"] == "08") & (clean["CMUN"] == "019")].copy()
print(f"Barcelona city: {len(bcn):,} observations, {bcn['CUDIS'].nunique()} districts")

# Madrid district names (official INE codes)
MADRID_DISTRICTS = {
    "2807901": "Centro", "2807902": "Arganzuela", "2807903": "Retiro",
    "2807904": "Salamanca", "2807905": "Chamartín", "2807906": "Tetuán",
    "2807907": "Chamberí", "2807908": "Fuencarral-\nEl Pardo",
    "2807909": "Moncloa-\nAravaca", "2807910": "Latina",
    "2807911": "Carabanchel", "2807912": "Usera", "2807913": "Puente de Vallecas",
    "2807914": "Moratalaz", "2807915": "Ciudad Lineal",
    "2807916": "Hortaleza", "2807917": "Villaverde",
    "2807918": "Villa de Vallecas", "2807919": "Vicálvaro",
    "2807920": "San Blas-\nCanillejas", "2807921": "Barajas",
}

BCN_DISTRICTS = {
    "0801901": "Ciutat Vella", "0801902": "Eixample",
    "0801903": "Sants-Montjuïc", "0801904": "Les Corts",
    "0801905": "Sarrià-\nSant Gervasi", "0801906": "Gràcia",
    "0801907": "Horta-\nGuinardó", "0801908": "Nou Barris",
    "0801909": "Sant Andreu", "0801910": "Sant Martí",
}

# ─── Compute district-level stats ───
def compute_district_stats(city_data, district_names):
    """Compute case distribution per district."""
    stats = city_data.groupby(["CUDIS", "case"]).size().unstack(fill_value=0)
    stats["total"] = stats.sum(axis=1)
    for c in CASE_ORDER:
        if c in stats.columns:
            stats[f"pct_{c}"] = stats[c] / stats["total"]
    stats["dominant"] = stats[CASE_ORDER].idxmax(axis=1)
    stats["name"] = stats.index.map(district_names)
    return stats

def compute_period_stats(city_data, district_names, year_range):
    """Compute case distribution for a specific time period."""
    period_data = city_data[city_data["year"].isin(year_range)]
    if len(period_data) == 0:
        return None
    stats = period_data.groupby(["CUDIS", "case"]).size().unstack(fill_value=0)
    for c in CASE_ORDER:
        if c not in stats.columns:
            stats[c] = 0
    stats["total"] = stats[CASE_ORDER].sum(axis=1)
    for c in CASE_ORDER:
        stats[f"pct_{c}"] = stats[c] / stats["total"]
    stats["dominant"] = stats[CASE_ORDER].idxmax(axis=1)
    stats["name"] = stats.index.map(district_names)
    return stats

madrid_stats = compute_district_stats(madrid, MADRID_DISTRICTS)
bcn_stats = compute_district_stats(bcn, BCN_DISTRICTS)

print("\nMadrid district case distribution:")
for cudis, row in madrid_stats.iterrows():
    name = row.get("name", cudis)
    dom = row["dominant"]
    pct_c = row.get("pct_C", 0)
    print(f"  {name}: dominant={dom}, C={pct_c*100:.1f}%")

print("\nBarcelona district case distribution:")
for cudis, row in bcn_stats.iterrows():
    name = row.get("name", cudis)
    dom = row["dominant"]
    pct_c = row.get("pct_C", 0)
    print(f"  {name}: dominant={dom}, C={pct_c*100:.1f}%")


# ─── FIGURE: Stacked bar charts by district (horizontal) ───
def plot_district_bars(stats, city_name, filename):
    """Plot horizontal stacked bar chart of case distribution by district."""
    stats_sorted = stats.sort_values("pct_C", ascending=True)
    names = [stats_sorted.loc[idx, "name"] or idx for idx in stats_sorted.index]
    n = len(names)

    fig, ax = plt.subplots(figsize=(14, max(8, n * 0.5)))
    left = np.zeros(n)

    for case in CASE_ORDER:
        col = f"pct_{case}"
        if col in stats_sorted.columns:
            vals = stats_sorted[col].values
            ax.barh(range(n), vals, left=left, color=CASE_COLORS[case],
                    label=CASE_LABELS[case], edgecolor="white", linewidth=0.5)
            left += vals

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Proportion", fontsize=13, fontweight="bold")
    ax.set_title(f"Gentrification Type Distribution by District: {city_name}\n(2012--2022)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(title="Case Type", title_fontsize=11, fontsize=10,
              loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=5, framealpha=0.95, edgecolor="#cccccc")
    ax.tick_params(axis="both", labelsize=11)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")

plot_district_bars(madrid_stats, "Madrid", "madrid_district_bars.png")
plot_district_bars(bcn_stats, "Barcelona", "barcelona_district_bars.png")


# ─── FIGURE: Temporal evolution by district (heatmap-style) ───
def plot_district_temporal(city_data, district_names, city_name, filename):
    """Show dominant case evolution over time per district."""
    periods = [
        ("2012-13", range(2012, 2014)),
        ("2014-15", range(2014, 2016)),
        ("2016-17", range(2016, 2018)),
        ("2018-19", range(2018, 2020)),
        ("2020-21", range(2020, 2022)),
    ]

    case_to_num = {"D": 0, "None": 1, "A": 2, "B": 3, "C": 4}
    num_to_case = {v: k for k, v in case_to_num.items()}

    districts = sorted(district_names.keys())
    district_labels = [district_names.get(d, d) for d in districts]
    matrix = np.full((len(districts), len(periods)), np.nan)

    for j, (label, years) in enumerate(periods):
        stats = compute_period_stats(city_data, district_names, years)
        if stats is None:
            continue
        for i, d in enumerate(districts):
            if d in stats.index:
                matrix[i, j] = case_to_num.get(stats.loc[d, "dominant"], 1)

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([CASE_COLORS[num_to_case[i]] for i in range(5)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, max(8, len(districts) * 0.5)))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels([p[0] for p in periods], fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(districts)))
    ax.set_yticklabels(district_labels, fontsize=11)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                case = num_to_case[int(matrix[i, j])]
                ax.text(j, i, case, ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if case in ["C", "A"] else "black")

    handles = [mpatches.Patch(facecolor=CASE_COLORS[c], edgecolor="#333",
                              label=CASE_LABELS[c]) for c in ["D", "None", "A", "B", "C"]]
    ax.legend(handles=handles, title="Dominant Case",
              title_fontsize=11, fontsize=10,
              loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=5, framealpha=0.95, edgecolor="#cccccc")

    ax.set_title(f"Temporal Evolution of Dominant Gentrification Type\n{city_name} Districts (2012--2021)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")

plot_district_temporal(madrid, MADRID_DISTRICTS, "Madrid", "madrid_temporal_heatmap.png")
plot_district_temporal(bcn, BCN_DISTRICTS, "Barcelona", "barcelona_temporal_heatmap.png")


# ─── FIGURE: Case C intensity per district (grouped bar: both cities) ───
print("Generating Case C comparison chart...")

# Combine both cities
madrid_c = madrid_stats[["name", "pct_C"]].copy()
madrid_c["city"] = "Madrid"
bcn_c = bcn_stats[["name", "pct_C"]].copy()
bcn_c["city"] = "Barcelona"
combined = pd.concat([madrid_c, bcn_c]).sort_values("pct_C", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=False)

for ax, (city, data, color) in zip(axes, [
    ("Madrid", madrid_stats.sort_values("pct_C"), "#264653"),
    ("Barcelona", bcn_stats.sort_values("pct_C"), "#264653"),
]):
    names = [data.loc[idx, "name"] or idx for idx in data.index]
    vals = data["pct_C"].values * 100
    n = len(names)

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n))
    ax.barh(range(n), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Case C Share (%)", fontsize=13, fontweight="bold")
    ax.set_title(city, fontsize=16, fontweight="bold", pad=10)
    ax.axvline(x=data["pct_C"].mean() * 100, color="#E63946",
               linestyle="--", linewidth=1.5, label=f"City avg: {data['pct_C'].mean()*100:.1f}%")
    ax.legend(fontsize=10, loc="lower right")

    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10, color="#333333")

    sns.despine(ax=ax, left=True, bottom=True)

fig.suptitle("Active Gentrification (Case C) Intensity by District",
             fontsize=18, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES_DIR / "district_case_c_comparison.png", bbox_inches="tight")
plt.close(fig)
print("  Saved district_case_c_comparison.png")


# ─── Print key stats for the paper ───
print("\n=== KEY STATS FOR PAPER ===")
print(f"\nMadrid city: {len(madrid):,} obs, {madrid['CUSEC'].nunique()} sections, {madrid['CUDIS'].nunique()} districts")
print(f"Barcelona city: {len(bcn):,} obs, {bcn['CUSEC'].nunique()} sections, {bcn['CUDIS'].nunique()} districts")

# Highest Case C districts in Madrid
print("\nMadrid - Highest Case C districts:")
mad_c_top = madrid_stats.nlargest(5, "pct_C")
for _, row in mad_c_top.iterrows():
    print(f"  {row['name']}: C={row['pct_C']*100:.1f}%, B={row.get('pct_B',0)*100:.1f}%, A={row.get('pct_A',0)*100:.1f}%")

print("\nMadrid - Lowest Case C districts:")
mad_c_bot = madrid_stats.nsmallest(5, "pct_C")
for _, row in mad_c_bot.iterrows():
    print(f"  {row['name']}: C={row['pct_C']*100:.1f}%, B={row.get('pct_B',0)*100:.1f}%, D={row.get('pct_D',0)*100:.1f}%")

print("\nBarcelona - Highest Case C districts:")
bcn_c_top = bcn_stats.nlargest(5, "pct_C")
for _, row in bcn_c_top.iterrows():
    print(f"  {row['name']}: C={row['pct_C']*100:.1f}%, B={row.get('pct_B',0)*100:.1f}%, A={row.get('pct_A',0)*100:.1f}%")

print("\nBarcelona - Lowest Case C districts:")
bcn_c_bot = bcn_stats.nsmallest(5, "pct_C")
for _, row in bcn_c_bot.iterrows():
    print(f"  {row['name']}: C={row['pct_C']*100:.1f}%, B={row.get('pct_B',0)*100:.1f}%, D={row.get('pct_D',0)*100:.1f}%")

# Temporal shifts
print("\nMadrid temporal dominant cases:")
for period_name, years in [("2012-13", range(2012, 2014)), ("2018-19", range(2018, 2020)), ("2020-21", range(2020, 2022))]:
    stats = compute_period_stats(madrid, MADRID_DISTRICTS, years)
    if stats is not None:
        dom_counts = stats["dominant"].value_counts()
        print(f"  {period_name}: {dict(dom_counts)}")

print("\nBarcelona temporal dominant cases:")
for period_name, years in [("2012-13", range(2012, 2014)), ("2018-19", range(2018, 2020)), ("2020-21", range(2020, 2022))]:
    stats = compute_period_stats(bcn, BCN_DISTRICTS, years)
    if stats is not None:
        dom_counts = stats["dominant"].value_counts()
        print(f"  {period_name}: {dict(dom_counts)}")

print("\nAll district maps generated.")
