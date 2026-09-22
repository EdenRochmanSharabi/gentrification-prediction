from pathlib import Path

BASE_DIR = Path("/Users/edenrochman/Documents/TFG")
DATA_DIR = BASE_DIR / "Base de datos"

IDEALISTA_PATH = DATA_DIR / "idealista_db_long.dta"
CARTOGRAPHY_PATH = (
    DATA_DIR
    / "cartografia_censo2011_nacional"
    / "SECC_CPV_E_20111101_01_R_INE.shp"
)
INFLATION_PATH = DATA_DIR / "Final EDA" / "Spain_Inflation_Rates_2012_2022.csv"

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MAPS_DIR = OUTPUT_DIR / "maps"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

PRICE_COLS = {
    "sale_price": "unitprice_residential_sale_all",
    "rent_price": "unitprice_residential_rent_all",
    "sale_stock": "stock_residential_sale_all",
    "rent_stock": "stock_residential_rent_all",
}

GEO_LEVELS = {
    "section": "CUSEC",
    "municipality": "NMUN",
    "province": "NPRO",
    "community": "NCA",
}
