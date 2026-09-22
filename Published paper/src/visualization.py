import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd

from config import MAPS_DIR, FIGURES_DIR

CASE_COLORS = {
    "A": "red",
    "B": "green",
    "C": "blue",
    "D": "purple",
    "None": "lightgrey",
    "Unknown": "white",
}

CASE_LEGEND = [
    Line2D([0], [0], marker="o", color="w", label="Case A (rent up)",
           markersize=10, markerfacecolor="red"),
    Line2D([0], [0], marker="o", color="w", label="Case B (sale up)",
           markersize=10, markerfacecolor="green"),
    Line2D([0], [0], marker="o", color="w", label="Case C (both up)",
           markersize=10, markerfacecolor="blue"),
    Line2D([0], [0], marker="o", color="w", label="Case D (both down)",
           markersize=10, markerfacecolor="purple"),
    Line2D([0], [0], marker="o", color="w", label="None",
           markersize=10, markerfacecolor="lightgrey"),
]


def _ensure_gdf(df):
    if not isinstance(df, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(df, geometry="geometry")
    return df


def _savefig(fig, directory, filename, dpi=300):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path


def plot_gentrification_cases(df, title, save_path=None, dpi=300):
    df = _ensure_gdf(df)
    if df.empty:
        print(f"No data for: {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = df["case"].map(CASE_COLORS)
    df.plot(ax=ax, color=colors, edgecolor="gray", linewidth=0.1)
    ax.set_aspect("equal")
    ax.set_xlim(df.total_bounds[[0, 2]])
    ax.set_ylim(df.total_bounds[[1, 3]])
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    ax.legend(handles=CASE_LEGEND, loc="upper right")

    if save_path:
        directory, filename = os.path.split(save_path)
        _savefig(fig, directory, filename, dpi)
    else:
        plt.show()


def plot_heatmap(df, geo_col, name, period, value_col, save_dir=None, dpi=300):
    df = _ensure_gdf(df)
    filtered = df[(df[geo_col] == name) & (df["period"] == period)]
    if filtered.empty:
        print(f"No data for {name} at {period}")
        return

    global_min = df[df[geo_col] == name][value_col].min()
    global_max = df[df[geo_col] == name][value_col].max()
    mean_val = filtered[value_col].mean()

    fig, ax = plt.subplots(figsize=(10, 10))
    filtered.plot(
        ax=ax, column=value_col, cmap="OrRd",
        vmin=global_min, vmax=global_max, legend=True,
        missing_kwds={"color": "lightgrey"},
        edgecolor="gray", linewidth=0.1,
    )
    ax.set_aspect("equal")
    ax.set_xlim(filtered.total_bounds[[0, 2]])
    ax.set_ylim(filtered.total_bounds[[1, 3]])
    ax.axis("off")
    ax.set_title(f"{value_col} — {name} ({period})")
    ax.annotate(f"Min: {global_min:.1f}", xy=(0.1, 0.1), xycoords="axes fraction",
                backgroundcolor="white")
    ax.annotate(f"Max: {global_max:.1f}", xy=(0.1, 0.05), xycoords="axes fraction",
                backgroundcolor="white")
    ax.annotate(f"Mean: {mean_val:.1f}", xy=(0.1, 0.0), xycoords="axes fraction",
                backgroundcolor="white")

    if save_dir:
        filename = f"heatmap_{name}_{period}.png"
        _savefig(fig, save_dir, filename, dpi)
    else:
        plt.show()


def plot_spain_heatmap(df, value_col, period, save_dir=None, dpi=300):
    df = _ensure_gdf(df)
    filtered = df[df["period"] == period]
    if filtered.empty:
        print(f"No data for period {period}")
        return

    global_min = df[value_col].min()
    global_max = df[value_col].max()
    mean_val = filtered[value_col].mean()

    fig, ax = plt.subplots(figsize=(10, 10))
    filtered.plot(
        ax=ax, column=value_col, cmap="OrRd",
        vmin=global_min, vmax=global_max, legend=True,
        missing_kwds={"color": "lightgrey"},
        edgecolor="gray", linewidth=0.1,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{value_col} — Spain ({period})")

    if save_dir:
        filename = f"spain_heatmap_{period}.png"
        _savefig(fig, save_dir, filename, dpi)
    else:
        plt.show()


def plot_combined_real_vs_predicted(df_real, df_pred, geo_col, name, period,
                                    save_dir=None, dpi=300):
    df_real = _ensure_gdf(df_real)
    df_pred = _ensure_gdf(df_pred)

    real = df_real[(df_real[geo_col] == name) & (df_real["period"] == period)]
    pred = df_pred[(df_pred[geo_col] == name) & (df_pred["period"] == period)]

    if real.empty or pred.empty:
        print(f"No data for {name} at {period}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, data, subtitle in [
        (axes[0], real, "Real"),
        (axes[1], pred, "Predicted (XGBoost)"),
    ]:
        colors = data["case"].map(CASE_COLORS)
        data.plot(ax=ax, color=colors, edgecolor="gray", linewidth=0.1)
        ax.set_aspect("equal")
        ax.set_xlim(data.total_bounds[[0, 2]])
        ax.set_ylim(data.total_bounds[[1, 3]])
        ax.axis("off")
        ax.set_title(f"{subtitle}: {name} ({period})", fontsize=14)
        ax.legend(handles=CASE_LEGEND, loc="upper right")

    if save_dir:
        filename = f"comparison_{name}_{period}.png"
        _savefig(fig, save_dir, filename, dpi)
    else:
        plt.show()


def plot_predictions_vs_actual(y_test, predictions, model_name, save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(y_test, predictions, alpha=0.5)
    axes[0].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        "k--", lw=2,
    )
    axes[0].set_title(f"{model_name} — Actual vs Predicted")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")

    residuals = y_test - predictions
    axes[1].hist(residuals, bins=30, alpha=0.7)
    axes[1].set_title(f"{model_name} — Residuals")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    if save_dir:
        filename = f"{model_name.lower().replace(' ', '_')}_evaluation.png"
        _savefig(fig, save_dir, filename)
    else:
        plt.show()


def plot_trend(df, group_col, value_col, title=None, top_n=5, save_dir=None):
    df = df.copy()
    df["period"] = pd.to_datetime(df["period"])
    pivot = df.pivot_table(index="period", columns=group_col, values=value_col, aggfunc="mean")
    top_groups = pivot.iloc[-1].nlargest(top_n).index

    cmap = plt.colormaps["nipy_spectral"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(pivot.columns))]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, col in enumerate(pivot.columns):
        alpha = 1.0 if col in top_groups else 0.3
        ax.plot(pivot.index, pivot[col], color=colors[i], alpha=alpha)
        if col in top_groups:
            ax.text(pivot.index[-1], pivot[col].iloc[-1], f" {col}", fontsize=8)

    ax.set_title(title or f"Trend: {value_col}")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_xlabel("Period")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_dir:
        filename = f"trend_{group_col}_{value_col}.png"
        _savefig(fig, save_dir, filename)
    else:
        plt.show()


