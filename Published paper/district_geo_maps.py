"""
Generate geographic district-level maps for Madrid and Barcelona
using OSM Overpass boundary data.
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
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import polygonize, unary_union

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

SCRATCHPAD = "/private/tmp/claude-501/-Users-edenrochman-Documents-Offline-projects-sandbox/1e1a11dd-6cf8-45cf-9ed8-b66bb18dad40/scratchpad"

# ─── Parse OSM Overpass data into GeoDataFrame ───
def parse_overpass_to_gdf(json_path, filter_names=None):
    with open(json_path) as f:
        data = json.load(f)

    elements = data["elements"]
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
    ways = {}
    for e in elements:
        if e["type"] == "way":
            node_refs = e.get("nodes", e.get("nd", []))
            coords = [nodes[n] for n in node_refs if n in nodes]
            if coords:
                ways[e["id"]] = coords

    records = []
    for e in elements:
        if e["type"] != "relation":
            continue
        tags = e.get("tags", {})
        name = tags.get("name", "")
        if filter_names and name not in filter_names:
            continue

        outer_rings = []
        for member in e.get("members", []):
            if member.get("role") == "outer" and member["type"] == "way":
                wid = member["ref"]
                if wid in ways:
                    outer_rings.append(ways[wid])

        if not outer_rings:
            continue

        lines = [LineString(coords) for coords in outer_rings if len(coords) >= 2]
        if not lines:
            continue

        merged = unary_union(lines)
        polys = list(polygonize(merged))

        if not polys:
            for coords in outer_rings:
                if len(coords) >= 4:
                    try:
                        p = Polygon(coords)
                        if p.is_valid and p.area > 0:
                            polys.append(p)
                    except:
                        pass

        if polys:
            geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
            records.append({"name": name, "geometry": geom})

    if not records:
        return gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


# ─── Load city data ───
print("Loading housing data...")
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

madrid = clean[(clean["CPRO"] == "28") & (clean["CMUN"] == "079")].copy()
bcn = clean[(clean["CPRO"] == "08") & (clean["CMUN"] == "019")].copy()

# District code to name mappings
MADRID_NAMES = {
    "2807901": "Centro", "2807902": "Arganzuela", "2807903": "Retiro",
    "2807904": "Salamanca", "2807905": "Chamartín", "2807906": "Tetuán",
    "2807907": "Chamberí", "2807908": "Fuencarral-El Pardo",
    "2807909": "Moncloa-Aravaca", "2807910": "Latina",
    "2807911": "Carabanchel", "2807912": "Usera",
    "2807913": "Puente de Vallecas", "2807914": "Moratalaz",
    "2807915": "Ciudad Lineal", "2807916": "Hortaleza",
    "2807917": "Villaverde", "2807918": "Villa de Vallecas",
    "2807919": "Vicálvaro", "2807920": "San Blas-Canillejas",
    "2807921": "Barajas",
}

BCN_NAMES = {
    "0801901": "Ciutat Vella", "0801902": "l'Eixample",
    "0801903": "Sants-Montjuïc", "0801904": "les Corts",
    "0801905": "Sarrià - Sant Gervasi", "0801906": "Gràcia",
    "0801907": "Horta-Guinardó", "0801908": "Nou Barris",
    "0801909": "Sant Andreu", "0801910": "Sant Martí",
}

# Map from OSM names to our CUDIS codes
MADRID_OSM_TO_CUDIS = {v: k for k, v in MADRID_NAMES.items()}
MADRID_OSM_TO_CUDIS["San Blas - Canillejas"] = "2807920"

BCN_OSM_TO_CUDIS = {v: k for k, v in BCN_NAMES.items()}

# ─── Parse geometries ───
print("Parsing Madrid districts...")
madrid_filter = set(MADRID_NAMES.values()) | {"San Blas - Canillejas"}
madrid_geo = parse_overpass_to_gdf(
    os.path.join(SCRATCHPAD, "madrid_overpass.json"),
    filter_names=madrid_filter
)
print(f"  Madrid: {len(madrid_geo)} district polygons")
for _, row in madrid_geo.iterrows():
    print(f"    {row['name']}: area={row.geometry.area:.6f}")

print("Parsing Barcelona districts...")
bcn_filter = set(BCN_NAMES.values())
bcn_geo = parse_overpass_to_gdf(
    os.path.join(SCRATCHPAD, "barcelona_overpass.json"),
    filter_names=bcn_filter
)
print(f"  Barcelona: {len(bcn_geo)} district polygons")
for _, row in bcn_geo.iterrows():
    print(f"    {row['name']}: area={row.geometry.area:.6f}")


# ─── Compute case stats per district ───
def get_district_case_pct(city_data, names_map, year_range=None):
    if year_range is not None:
        city_data = city_data[city_data["year"].isin(year_range)]
    stats = city_data.groupby(["CUDIS", "case"]).size().unstack(fill_value=0)
    for c in CASE_ORDER:
        if c not in stats.columns:
            stats[c] = 0
    stats["total"] = stats[CASE_ORDER].sum(axis=1)
    for c in CASE_ORDER:
        stats[f"pct_{c}"] = stats[c] / stats["total"]
    stats["dominant"] = stats[CASE_ORDER].idxmax(axis=1)
    stats["name"] = stats.index.map(names_map)
    return stats


def merge_geo_with_stats(geo, stats, osm_to_cudis):
    geo = geo.copy()
    geo["CUDIS"] = geo["name"].map(osm_to_cudis)
    merged = geo.merge(stats, left_on="CUDIS", right_index=True,
                       how="left", suffixes=("", "_stats"))
    return merged


# ─── Generate maps ───
def plot_city_case_map(geo, city_data, names_map, osm_to_cudis,
                       city_name, filename, year_range=None,
                       metric="pct_C", cmap="Blues", label="Case C Share"):
    stats = get_district_case_pct(city_data, names_map, year_range)
    merged = merge_geo_with_stats(geo, stats, osm_to_cudis)

    if len(merged) == 0:
        print(f"  WARNING: no merged data for {city_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    vmin = merged[metric].min() if metric in merged.columns else 0
    vmax = merged[metric].max() if metric in merged.columns else 1

    merged.plot(column=metric, ax=ax, cmap=cmap, edgecolor="#333333",
                linewidth=1.2, legend=True, vmin=vmin, vmax=vmax,
                legend_kwds={"label": label, "orientation": "horizontal",
                             "pad": 0.02, "shrink": 0.6})

    for _, row in merged.iterrows():
        if row.geometry and not row.geometry.is_empty:
            centroid = row.geometry.centroid
            district_name = row.get("name", "")
            if district_name:
                short = district_name.replace("Fuencarral-El Pardo", "Fuencarral")
                short = short.replace("Puente de Vallecas", "P. Vallecas")
                short = short.replace("Villa de Vallecas", "V. Vallecas")
                short = short.replace("San Blas-Canillejas", "San Blas")
                short = short.replace("San Blas - Canillejas", "San Blas")
                short = short.replace("Sarrià - Sant Gervasi", "Sarrià-St.G.")
                short = short.replace("Sants-Montjuïc", "Sants-Montj.")
                short = short.replace("Horta-Guinardó", "Horta-Guin.")
                short = short.replace("Ciudad Lineal", "C. Lineal")
                short = short.replace("Moncloa-Aravaca", "Moncloa")
                val = row.get(metric, 0)
                ax.annotate(f"{short}\n{val*100:.0f}%",
                           xy=(centroid.x, centroid.y),
                           ha="center", va="center",
                           fontsize=7, fontweight="bold",
                           color="white" if val > (vmin + vmax) / 2 else "black",
                           bbox=dict(boxstyle="round,pad=0.15",
                                    facecolor="white", alpha=0.6,
                                    edgecolor="none"))

    period_str = f" ({min(year_range)}--{max(year_range)})" if year_range else " (2012--2022)"
    ax.set_title(f"{label}: {city_name} Districts{period_str}",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_city_dominant_map(geo, city_data, names_map, osm_to_cudis,
                           city_name, filename, year_range=None):
    stats = get_district_case_pct(city_data, names_map, year_range)
    merged = merge_geo_with_stats(geo, stats, osm_to_cudis)

    if len(merged) == 0:
        print(f"  WARNING: no merged data for {city_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    for case_val, color in CASE_COLORS.items():
        subset = merged[merged["dominant"] == case_val]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor="#333333", linewidth=1.5)

    for _, row in merged.iterrows():
        if row.geometry and not row.geometry.is_empty:
            centroid = row.geometry.centroid
            district_name = row.get("name", "")
            dom = row.get("dominant", "")
            if district_name:
                short = district_name.replace("Fuencarral-El Pardo", "Fuencarral")
                short = short.replace("Puente de Vallecas", "P. Vallecas")
                short = short.replace("Villa de Vallecas", "V. Vallecas")
                short = short.replace("San Blas-Canillejas", "San Blas")
                short = short.replace("San Blas - Canillejas", "San Blas")
                short = short.replace("Sarrià - Sant Gervasi", "Sarrià-St.G.")
                short = short.replace("Sants-Montjuïc", "Sants-Montj.")
                short = short.replace("Horta-Guinardó", "Horta-Guin.")
                short = short.replace("Ciudad Lineal", "C. Lineal")
                short = short.replace("Moncloa-Aravaca", "Moncloa")
                lum = {"C": True, "A": True, "B": False, "D": False, "None": False}
                text_color = "white" if lum.get(dom, False) else "black"
                ax.annotate(short,
                           xy=(centroid.x, centroid.y),
                           ha="center", va="center",
                           fontsize=7.5, fontweight="bold",
                           color=text_color)

    handles = [mpatches.Patch(facecolor=CASE_COLORS[c], edgecolor="#333",
                              label=CASE_LABELS[c]) for c in CASE_ORDER]
    ax.legend(handles=handles, title="Dominant Case Type",
              title_fontsize=12, fontsize=10,
              loc="lower left", framealpha=0.95, edgecolor="#cccccc")

    period_str = f" ({min(year_range)}--{max(year_range)})" if year_range else " (2012--2022)"
    ax.set_title(f"Dominant Gentrification Type: {city_name} Districts{period_str}",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_combined_temporal_dominant(geo_mad, geo_bcn, madrid, bcn,
                                    mad_names, bcn_names,
                                    mad_osm, bcn_osm, filename):
    """Side-by-side temporal comparison: early vs late period, both cities."""
    periods = [
        ("2012--2013", range(2012, 2014)),
        ("2018--2019", range(2018, 2020)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    for col_idx, (period_name, years) in enumerate(periods):
        for row_idx, (geo, city_data, names_map, osm_map, city_name) in enumerate([
            (geo_mad, madrid, mad_names, mad_osm, "Madrid"),
            (geo_bcn, bcn, bcn_names, bcn_osm, "Barcelona"),
        ]):
            ax = axes[row_idx, col_idx]
            stats = get_district_case_pct(city_data, names_map, years)
            merged = merge_geo_with_stats(geo, stats, osm_map)

            if len(merged) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            for case_val, color in CASE_COLORS.items():
                subset = merged[merged["dominant"] == case_val]
                if len(subset) > 0:
                    subset.plot(ax=ax, color=color, edgecolor="#333333", linewidth=1.0)

            for _, row in merged.iterrows():
                if row.geometry and not row.geometry.is_empty:
                    centroid = row.geometry.centroid
                    district_name = row.get("name", "")
                    dom = row.get("dominant", "")
                    if district_name:
                        short = district_name.split("-")[0].split(" ")[0]
                        if len(short) > 8:
                            short = short[:7] + "."
                        lum = {"C": True, "A": True, "B": False, "D": False, "None": False}
                        text_color = "white" if lum.get(dom, False) else "black"
                        ax.annotate(short,
                                   xy=(centroid.x, centroid.y),
                                   ha="center", va="center",
                                   fontsize=6, fontweight="bold",
                                   color=text_color)

            ax.set_title(f"{city_name} ({period_name})",
                         fontsize=15, fontweight="bold", pad=8)
            ax.set_axis_off()

    handles = [mpatches.Patch(facecolor=CASE_COLORS[c], edgecolor="#333",
                              label=CASE_LABELS[c]) for c in CASE_ORDER]
    fig.legend(handles=handles, title="Dominant Case Type",
               title_fontsize=13, fontsize=12,
               loc="lower center", ncol=5, framealpha=0.95,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Temporal Evolution of Dominant Gentrification Type\nby District",
                 fontsize=20, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


if len(madrid_geo) > 0 and len(bcn_geo) > 0:
    # Case C intensity maps
    print("\nGenerating Case C intensity maps...")
    plot_city_case_map(madrid_geo, madrid, MADRID_NAMES, MADRID_OSM_TO_CUDIS,
                       "Madrid", "madrid_case_c_map.png",
                       metric="pct_C", cmap="Blues", label="Active Gentrification (Case C) Share")
    plot_city_case_map(bcn_geo, bcn, BCN_NAMES, BCN_OSM_TO_CUDIS,
                       "Barcelona", "barcelona_case_c_map.png",
                       metric="pct_C", cmap="Blues", label="Active Gentrification (Case C) Share")

    # Dominant case maps (full period)
    print("\nGenerating dominant case maps...")
    plot_city_dominant_map(madrid_geo, madrid, MADRID_NAMES, MADRID_OSM_TO_CUDIS,
                           "Madrid", "madrid_dominant_map.png")
    plot_city_dominant_map(bcn_geo, bcn, BCN_NAMES, BCN_OSM_TO_CUDIS,
                           "Barcelona", "barcelona_dominant_map.png")

    # Dominant case maps by period
    print("\nGenerating temporal dominant maps...")
    for period_name, years, suffix in [
        ("early", range(2012, 2016), "2012_2015"),
        ("late", range(2018, 2022), "2018_2021"),
    ]:
        plot_city_dominant_map(madrid_geo, madrid, MADRID_NAMES, MADRID_OSM_TO_CUDIS,
                               "Madrid", f"madrid_dominant_{suffix}.png", year_range=years)
        plot_city_dominant_map(bcn_geo, bcn, BCN_NAMES, BCN_OSM_TO_CUDIS,
                               "Barcelona", f"barcelona_dominant_{suffix}.png", year_range=years)

    # Combined temporal figure
    print("\nGenerating combined temporal figure...")
    plot_combined_temporal_dominant(madrid_geo, bcn_geo, madrid, bcn,
                                    MADRID_NAMES, BCN_NAMES,
                                    MADRID_OSM_TO_CUDIS, BCN_OSM_TO_CUDIS,
                                    "cities_temporal_dominant.png")

    print("\nAll geographic district maps generated.")
else:
    print("ERROR: Could not parse district geometries")
