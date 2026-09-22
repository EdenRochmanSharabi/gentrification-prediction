"""
Generate regional analysis figures and compute statistics for the paper.
Province-level analysis of WHERE each gentrification type occurs.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
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
MAINLAND_EXCLUDE = ["Las Palmas", "Santa Cruz de Tenerife"]

# ─── Load data ───
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

clean = df[df["case"].isin(["A", "B", "C", "D", "None"])].copy()
print(f"Data loaded: {len(clean):,} classified observations")

# ─── Province-level statistics ───
print("\n=== PROVINCE-LEVEL CASE STATISTICS ===\n")

prov_case = clean.groupby(["NPRO", "case"]).size().unstack(fill_value=0)
prov_case["total"] = prov_case.sum(axis=1)
for c in ["A", "B", "C", "D", "None"]:
    if c in prov_case.columns:
        prov_case[f"pct_{c}"] = prov_case[c] / prov_case["total"]

prov_case["dominant"] = prov_case[["A", "B", "C", "D", "None"]].idxmax(axis=1)

# Print top provinces for each case type
for case in ["A", "B", "C", "D"]:
    col = f"pct_{case}"
    top5 = prov_case.nlargest(5, col)[[col, case, "total"]]
    print(f"Top 5 provinces for Case {case}:")
    for prov, row in top5.iterrows():
        print(f"  {prov}: {row[col]*100:.1f}% ({int(row[case]):,} obs)")
    # Also bottom 5
    bot5 = prov_case.nsmallest(5, col)[[col, case, "total"]]
    print(f"Bottom 5 provinces for Case {case}:")
    for prov, row in bot5.iterrows():
        print(f"  {prov}: {row[col]*100:.1f}% ({int(row[case]):,} obs)")
    print()

# National averages
national = clean.groupby("case").size()
total = national.sum()
print("National case distribution:")
for c in CASE_ORDER:
    if c in national.index:
        print(f"  {c}: {national[c]:,} ({national[c]/total*100:.1f}%)")

# Dominant case by province
print(f"\nDominant case by province:")
dom_counts = prov_case["dominant"].value_counts()
for case, count in dom_counts.items():
    provs = prov_case[prov_case["dominant"] == case].index.tolist()
    print(f"  {case}: {count} provinces - {', '.join(provs[:8])}{'...' if len(provs) > 8 else ''}")

# Test period analysis (2020+)
test_data = clean[clean["year"] >= 2020].copy()
prov_test = test_data.groupby(["NPRO", "case"]).size().unstack(fill_value=0)
prov_test["total"] = prov_test.sum(axis=1)
for c in ["A", "B", "C", "D", "None"]:
    if c in prov_test.columns:
        prov_test[f"pct_{c}"] = prov_test[c] / prov_test["total"]
prov_test["dominant"] = prov_test[["A", "B", "C", "D", "None"]].idxmax(axis=1)

print(f"\n=== TEST PERIOD (2020+) DOMINANT CASE ===")
dom_test = prov_test["dominant"].value_counts()
for case, count in dom_test.items():
    provs = prov_test[prov_test["dominant"] == case].index.tolist()
    print(f"  {case}: {count} provinces")
    for p in provs:
        print(f"    - {p}")

# Early vs late period shift
print(f"\n=== TEMPORAL SHIFT: EARLY (2012-2015) vs LATE (2018-2021) ===")
for period_name, year_range in [("2012-2015", range(2012, 2016)), ("2018-2021", range(2018, 2022))]:
    period_data = clean[clean["year"].isin(year_range)]
    period_dom = period_data.groupby(["NPRO", "case"]).size().unstack(fill_value=0)
    period_dom["dominant"] = period_dom.idxmax(axis=1)
    dc = period_dom["dominant"].value_counts()
    print(f"\n{period_name}:")
    for case, count in dc.items():
        print(f"  {case}: {count} provinces")

# Geographic clusters
print(f"\n=== GEOGRAPHIC CLUSTERS ===")
coastal_med = ["Barcelona", "Girona", "Tarragona", "Valencia/Valéncia",
               "Alicante/Alacant", "Murcia", "Balears, Illes", "Málaga", "Almería"]
interior = ["Madrid", "Toledo", "Ciudad Real", "Cuenca", "Guadalajara",
            "Segovia", "Ávila", "Valladolid", "Salamanca", "Zamora"]
north = ["Bizkaia", "Gipuzkoa", "Araba/Álava", "Navarra", "Cantabria", "Asturias"]
south = ["Sevilla", "Córdoba", "Jaén", "Granada", "Cádiz", "Huelva"]

for region_name, provinces in [("Mediterranean coast", coastal_med),
                                ("Interior/Central", interior),
                                ("Northern Spain", north),
                                ("Andalusia", south)]:
    region_data = clean[clean["NPRO"].isin(provinces)]
    if len(region_data) == 0:
        continue
    region_cases = region_data.groupby("case").size()
    region_total = region_cases.sum()
    print(f"\n{region_name} ({len(region_data):,} obs):")
    for c in CASE_ORDER:
        if c in region_cases.index:
            print(f"  {c}: {region_cases[c]/region_total*100:.1f}%")

# ─── Load geodata ───
print("\n\nGenerating maps...")
ne_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
spain_geo = gpd.read_file(ne_url)
spain_geo = spain_geo[spain_geo["admin"] == "Spain"].copy()

name_map = {
    "Álava": "Araba/Álava", "Alicante": "Alicante/Alacant",
    "Castellón": "Castellón/Castelló", "Valencia": "Valencia/Valéncia",
    "La Coruña": "Coruña, A", "Gerona": "Girona", "Lérida": "Lleida",
    "Orense": "Ourense", "Vizcaya": "Bizkaia", "Guipúzcoa": "Gipuzkoa",
    "Baleares": "Balears, Illes", "Las Palmas": "Palmas, Las",
    "Santa Cruz de Tenerife": "Santa Cruz de Tenerife",
}
spain_geo["NPRO_match"] = spain_geo["name"].map(name_map).fillna(spain_geo["name"])

# ─── FIGURE: 2x2 panel of all case type intensities ───
print("Generating 2x2 case intensity panel...")
spain_merged = spain_geo.merge(prov_case, left_on="NPRO_match", right_index=True, how="left")
mainland = spain_merged[~spain_merged["name"].isin(MAINLAND_EXCLUDE)].copy()

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
case_info = [
    ("A", "Renter Pressure", "Reds"),
    ("B", "Speculative Investment", "Greens"),
    ("C", "Active Gentrification", "Blues"),
    ("D", "Neighborhood Degradation", "YlOrBr"),
]

for ax, (case, title, cmap) in zip(axes.flat, case_info):
    col = f"pct_{case}"
    vmax = mainland[col].quantile(0.95) if col in mainland.columns else 0.3
    mainland.plot(column=col, ax=ax, cmap=cmap, edgecolor="#333333",
                  linewidth=0.5, legend=True, vmin=0, vmax=vmax,
                  missing_kwds={"color": "#f0f0f0", "edgecolor": "#999"},
                  legend_kwds={"orientation": "horizontal", "pad": 0.02,
                               "shrink": 0.6, "label": f"Proportion of Case {case}"})
    ax.set_title(f"Case {case}: {title}", fontsize=16, fontweight="bold", pad=10)
    ax.set_axis_off()

fig.suptitle("Geographic Distribution of Gentrification Types by Province\n(2012--2022)",
             fontsize=20, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES_DIR / "spain_all_cases_intensity_panel.png", bbox_inches="tight")
plt.close(fig)
print("  Saved spain_all_cases_intensity_panel.png")

# ─── FIGURE: Dominant case map (full period) ───
print("Generating dominant case map...")
fig, ax = plt.subplots(figsize=(14, 10))
mainland_dom = spain_geo.merge(prov_case[["dominant"]], left_on="NPRO_match",
                                right_index=True, how="left")
mainland_dom = mainland_dom[~mainland_dom["name"].isin(MAINLAND_EXCLUDE)]

for case_val, color in CASE_COLORS.items():
    subset = mainland_dom[mainland_dom["dominant"] == case_val]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor="#333333", linewidth=0.5)

no_data = mainland_dom[mainland_dom["dominant"].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color="#f0f0f0", edgecolor="#999999", linewidth=0.5)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CASE_COLORS[c], edgecolor="#333")
           for c in CASE_ORDER]
labels = [CASE_LABELS[c] for c in CASE_ORDER]
ax.legend(handles, labels, title="Dominant Case Type", title_fontsize=13,
          fontsize=12, loc="lower left", framealpha=0.95, edgecolor="#cccccc")
ax.set_title("Dominant Gentrification Type by Province (2012--2022)",
             fontsize=16, fontweight="bold", pad=15)
ax.set_axis_off()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "spain_dominant_case_map.png", bbox_inches="tight")
plt.close(fig)
print("  Saved spain_dominant_case_map.png")

# ─── FIGURE: Top provinces bar chart ───
print("Generating top provinces bar chart...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
for ax, (case, title) in zip(axes.flat, [
    ("A", "Case A: Renter Pressure"),
    ("B", "Case B: Speculative Investment"),
    ("C", "Case C: Active Gentrification"),
    ("D", "Case D: Degradation"),
]):
    col = f"pct_{case}"
    top10 = prov_case.nlargest(10, col)
    colors = [CASE_COLORS[case]] * 10
    ax.barh(range(10), top10[col].values * 100, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10.index, fontsize=11)
    ax.set_xlabel("Share (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.invert_yaxis()
    for i, (_, row) in enumerate(top10.iterrows()):
        ax.text(row[col] * 100 + 0.3, i, f"{row[col]*100:.1f}%",
                va="center", fontsize=10, color="#333333")
    sns.despine(ax=ax, left=True, bottom=True)

fig.suptitle("Top 10 Provinces by Gentrification Type Intensity",
             fontsize=18, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES_DIR / "top_provinces_by_case.png", bbox_inches="tight")
plt.close(fig)
print("  Saved top_provinces_by_case.png")

print("\nAll regional analysis figures generated.")
