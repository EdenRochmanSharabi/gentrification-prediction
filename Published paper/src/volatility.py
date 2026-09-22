import numpy as np
import pandas as pd


def bollinger_bands(df, group_col, value_col, window=20, num_std=2):
    """Compute average Bollinger Band width per group as a volatility measure."""
    df = df.copy().reset_index(drop=True)

    def _compute(group):
        group["moving_avg"] = group[value_col].rolling(window=window, min_periods=1).mean()
        group["moving_std"] = group[value_col].rolling(window=window, min_periods=1).std(ddof=0)
        group["upper"] = group["moving_avg"] + num_std * group["moving_std"]
        group["lower"] = group["moving_avg"] - num_std * group["moving_std"]
        group["bollinger_width"] = group["upper"] - group["lower"]
        return group

    computed = df.groupby(group_col, group_keys=False).apply(_compute)
    volatility = (
        computed.groupby(group_col)["bollinger_width"]
        .mean()
        .reset_index()
        .sort_values("bollinger_width", ascending=False)
    )
    return volatility


def campbell_shiller(df, group_col, value_col, long_window=24, short_window=2):
    """Cyclically adjusted price deviation metric per group."""
    df = df.copy().reset_index(drop=True)

    def _compute(group):
        group["long_avg"] = group[value_col].rolling(window=long_window, min_periods=1).mean()
        group["short_avg"] = group[value_col].rolling(window=short_window, min_periods=1).mean()
        group["long_std"] = group[value_col].rolling(window=long_window, min_periods=1).std(ddof=0)
        safe_std = group["long_std"].replace(0, np.nan)
        group["adjusted_deviation"] = (group["short_avg"] - group["long_avg"]) / safe_std
        return group

    computed = df.groupby(group_col, group_keys=False).apply(_compute)
    volatility = (
        computed.groupby(group_col)["adjusted_deviation"]
        .mean()
        .reset_index()
        .sort_values("adjusted_deviation", ascending=False)
    )
    return volatility


def bollinger_width_at_date(df, value_col, window=20, num_std=2):
    """Compute pointwise Bollinger width (for heatmap at a specific date)."""
    df = df.copy()
    roll_std = df[value_col].rolling(window=window).std()
    df["bollinger_width"] = num_std * 2 * roll_std
    return df
