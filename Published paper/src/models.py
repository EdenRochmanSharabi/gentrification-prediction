import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import MODELS_DIR


def build_xgb_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        XGBRegressor(objective="reg:squarederror"),
    )


def build_rf_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        RandomForestRegressor(random_state=42),
    )


def build_lasso_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        Lasso(max_iter=10000),
    )


def build_voting_regressor():
    return VotingRegressor(
        estimators=[
            ("xgb", build_xgb_pipeline()),
            ("rf", build_rf_pipeline()),
            ("lasso", build_lasso_pipeline()),
        ]
    )


XGB_PARAM_GRID = {
    "xgbregressor__max_depth": [3, 5, 7],
    "xgbregressor__n_estimators": [100, 200, 300],
    "xgbregressor__learning_rate": [0.01, 0.05, 0.1],
    "xgbregressor__colsample_bytree": [0.3, 0.5, 0.7],
    "xgbregressor__subsample": [0.7, 0.9, 1.0],
    "xgbregressor__alpha": [0, 10, 20],
}


def train_with_gridsearch(pipeline, param_grid, X_train, y_train, cv=3):
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-gs.best_score_):.4f}")
    return gs


def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name} — RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"name": name, "rmse": rmse, "r2": r2, "predictions": preds}


def prepare_splits(df, nmun, test_size=0.2):
    local = df[df["NMUN"] == nmun].copy()
    price_cols = [
        "unitprice_residential_sale_all",
        "unitprice_residential_rent_all",
    ]
    local = local.dropna(subset=price_cols).sort_values("period")
    train, test = train_test_split(local, test_size=test_size, shuffle=False)
    return train, test


def save_model(model, name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Saved: {path}")
    return path


def load_model(name):
    path = MODELS_DIR / f"{name}.pkl"
    return joblib.load(path)
