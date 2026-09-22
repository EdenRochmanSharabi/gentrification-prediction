import pandas as pd
import geopandas as gpd
from dbfread import DBF

from config import IDEALISTA_PATH, CARTOGRAPHY_PATH


def load_idealista(path=IDEALISTA_PATH):
    df = pd.read_stata(path)
    df["locationid"] = df["locationid"].apply(lambda x: f"{int(x):010d}")
    return df


def load_cartography(path=CARTOGRAPHY_PATH):
    gdf = gpd.read_file(path)
    gdf = gdf.drop(columns=["OBJECTID"], errors="ignore")
    return gdf


def load_cartography_metadata(path=CARTOGRAPHY_PATH):
    """Load geographic metadata from the .dbf file (no geometry needed)."""
    dbf_path = str(path).replace(".shp", ".dbf")
    table = DBF(dbf_path)
    df = pd.DataFrame(iter(table))
    df = df.drop(columns=["OBJECTID"], errors="ignore")
    return df


def merge_datasets(df, gdf):
    joined = pd.merge(df, gdf, how="outer", left_on="locationid", right_on="CUSEC")
    joined["period"] = pd.to_datetime(joined["period"])
    joined["NPRO"] = joined["NPRO"].replace("Gipuzcoa", "Gipuzkoa")
    joined["NCA"] = joined["NCA"].replace("Pais Vasco", "País Vasco")
    if "geometry" in joined.columns:
        joined = gpd.GeoDataFrame(joined, geometry="geometry")
    return joined


def load_data():
    df = load_idealista()
    shp_path = CARTOGRAPHY_PATH
    if shp_path.exists():
        gdf = load_cartography(shp_path)
    else:
        gdf = load_cartography_metadata(shp_path)
    return merge_datasets(df, gdf)


def clean_data(df):
    price_cols = [
        "stock_residential_sale_all",
        "unitprice_residential_sale_all",
        "stock_residential_rent_all",
        "unitprice_residential_rent_all",
    ]
    return df.dropna(subset=price_cols)
