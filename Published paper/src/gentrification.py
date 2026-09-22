"""
Gentrification case classification based on the thesis operationalization.

Cases (based on % change in rent and sale prices per census section):
  A: rent up,   sale stable/down  → pressure from renters
  B: sale up,   rent stable/down  → speculative purchases / pioneer gentrifiers
  C: both up                      → gentrification in progress
  D: both down                    → degradation / abandonment (pre-gentrification)
  None: no significant change
"""

import numpy as np
import pandas as pd
import geopandas as gpd

SALE_COL = "unitprice_residential_sale_all"
RENT_COL = "unitprice_residential_rent_all"


def classify_cases(df):
    """Classify gentrification cases from pre-computed rent_change / sale_change columns.

    The conditions are evaluated in order of specificity:
      C requires BOTH positive — checked first to avoid A or B stealing it.
      D requires BOTH negative — checked second.
      A and B are the single-direction cases.
    Rows with NaN in rent_change or sale_change are classified as "Unknown".
    """
    df = df.copy()
    has_data = df["rent_change"].notna() & df["sale_change"].notna()
    conditions = [
        ~has_data,
        has_data & (df["rent_change"] > 0) & (df["sale_change"] > 0),   # C
        has_data & (df["rent_change"] < 0) & (df["sale_change"] < 0),   # D
        has_data & (df["rent_change"] > 0) & (df["sale_change"] <= 0),  # A
        has_data & (df["sale_change"] > 0) & (df["rent_change"] <= 0),  # B
    ]
    choices = ["Unknown", "C", "D", "A", "B"]
    df["case"] = np.select(conditions, choices, default="None")
    return df


def compute_pct_change(df, group_col="CUSEC"):
    """Compute per-group percentage change in rent and sale prices.

    BUG FIX: the original code computed pct_change() across the entire DataFrame
    instead of within each CUSEC group, mixing unrelated census sections.
    """
    df = df.copy()
    df = df.sort_values(["period"])
    grouped = df.groupby(group_col)
    df["rent_change"] = grouped[RENT_COL].pct_change()
    df["sale_change"] = grouped[SALE_COL].pct_change()
    return df


def annual_changes(df, nmun=None, group_col="CUSEC"):
    """Aggregate price changes by year for each census section."""
    df = df.copy()
    if nmun:
        df = df[df["NMUN"] == nmun]

    df["year"] = df["period"].dt.year
    df = compute_pct_change(df, group_col)

    annual_dfs = {}
    for year, year_df in df.groupby("year"):
        agg = (
            year_df.groupby(group_col)
            .agg({"rent_change": "sum", "sale_change": "sum", "geometry": "first"})
            .reset_index()
        )
        agg = classify_cases(agg)
        annual_dfs[year] = gpd.GeoDataFrame(agg, geometry="geometry")
    return annual_dfs


def cumulative_changes(df, nmun=None, group_col="CUSEC"):
    """Compute cumulative price changes over the full period for each section."""
    df = df.copy()
    if nmun:
        df = df[df["NMUN"] == nmun]

    df = compute_pct_change(df, group_col)

    cumulative = (
        df.groupby(group_col)
        .agg({"rent_change": "sum", "sale_change": "sum", "geometry": "first"})
        .reset_index()
    )
    cumulative = classify_cases(cumulative)
    return gpd.GeoDataFrame(cumulative, geometry="geometry")


def check_all_cases(df, start_date, end_date, nmun=None, group_col="CUSEC"):
    """Find sections matching each gentrification case in a date range.

    BUG FIX: uses per-group pct_change instead of global pct_change.
    """
    df = df.copy()
    filtered = df[(df["period"] >= start_date) & (df["period"] <= end_date)]
    if nmun:
        filtered = filtered[filtered["NMUN"] == nmun]

    filtered = compute_pct_change(filtered, group_col)

    return {
        "Case A": filtered[(filtered["rent_change"] > 0) & (filtered["sale_change"] <= 0)],
        "Case B": filtered[(filtered["sale_change"] > 0) & (filtered["rent_change"] <= 0)],
        "Case C": filtered[(filtered["rent_change"] > 0) & (filtered["sale_change"] > 0)],
        "Case D": filtered[(filtered["sale_change"] < 0) & (filtered["rent_change"] < 0)],
    }
