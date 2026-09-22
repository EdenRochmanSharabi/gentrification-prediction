import pandas as pd


def add_temporal_features(df):
    df = df.copy()
    df["month"] = df["period"].dt.month
    df["quarter"] = df["period"].dt.quarter
    df["year"] = df["period"].dt.year
    return df


def add_lag_features(df, group_col="CUSEC"):
    df = df.copy()
    df = df.sort_values(["period"])

    sale_col = "unitprice_residential_sale_all"
    rent_col = "unitprice_residential_rent_all"

    grouped = df.groupby(group_col)
    df["price_lag_1"] = grouped[sale_col].shift(1)
    df["price_roll_mean_3"] = (
        grouped[sale_col]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rent_price_lag_1"] = grouped[rent_col].shift(1)
    df["rent_price_roll_mean_3"] = (
        grouped[rent_col]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def add_all_features(df, group_col="CUSEC"):
    df = add_temporal_features(df)
    df = add_lag_features(df, group_col)
    return df


FEATURE_COLS_SALE = [
    "stock_residential_sale_all",
    "stock_residential_rent_all",
    "unitprice_residential_rent_all",
    "month",
    "quarter",
    "year",
    "price_lag_1",
    "price_roll_mean_3",
]

FEATURE_COLS_RENT = [
    "stock_residential_sale_all",
    "unitprice_residential_sale_all",
    "stock_residential_rent_all",
    "month",
    "quarter",
    "year",
    "rent_price_lag_1",
    "rent_price_roll_mean_3",
]

TARGET_SALE = "unitprice_residential_sale_all"
TARGET_RENT = "unitprice_residential_rent_all"
