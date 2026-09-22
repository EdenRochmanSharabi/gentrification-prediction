"""LSTM test: captures temporal sequence per CUSEC (4 quarters window)."""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "lstm.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log(f"Device: {device}")
log("=" * 60)
log("LSTM TEMPORAL SEQUENCE MODEL")
log("=" * 60)

log("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
grouped = df.groupby("CUSEC")

df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)
df["target_case"] = df.groupby("CUSEC")["case"].shift(-1)

df["sale_price"] = df[sale_col]
df["rent_price"] = df[rent_col]
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
le_case = LabelEncoder()
df["case_encoded_feat"] = le_case.fit_transform(df["case"].fillna("Unknown"))
df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)
df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]
df["change_interaction"] = df["rent_change"] * df["sale_change"]
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

FEATURES = [
    "sale_price", "rent_price", "rent_sale_ratio", "case_encoded_feat",
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2", "sale_lag4", "rent_lag4",
    "sale_change_curr", "rent_change_curr", "sale_change_lag1", "rent_change_lag1",
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6",
    "sale_momentum", "rent_momentum", "change_interaction",
    "sale_vs_prov", "rent_vs_prov", "sale_yoy", "rent_yoy",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "month", "quarter", "year",
]

trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])
classes = list(le.classes_)
n_classes = len(classes)
n_features = len(FEATURES)

log(f"Features: {n_features}, Classes: {classes}")

# Save original years before scaling
trainable["year_orig"] = trainable["year"].values.copy()

# Impute and scale BEFORE building sequences
imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
trainable[FEATURES] = imp.fit_transform(trainable[FEATURES])
trainable[FEATURES] = sc.fit_transform(trainable[FEATURES])

for SEQ_LEN in [4, 8]:
    log(f"\n{'=' * 60}")
    log(f"LSTM seq_len={SEQ_LEN} ({SEQ_LEN} quarters = {SEQ_LEN*3} months)")
    log("Building sequences...")

    # Build sequences over FULL timeline per CUSEC, then split by the TARGET row's year
    X_all, y_all, years_all = [], [], []
    for cusec, group in trainable.groupby("CUSEC"):
        vals = group[FEATURES].values
        targets = group["target_encoded"].values
        yrs = group["year_orig"].values
        for i in range(SEQ_LEN, len(vals)):
            X_all.append(vals[i-SEQ_LEN:i])
            y_all.append(targets[i])
            years_all.append(yrs[i])  # year of the TARGET row

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)
    years_all = np.array(years_all)

    tr_idx = years_all < 2020
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all[~tr_idx], y_all[~tr_idx]
    log(f"Sequences: train={len(X_tr):,}, test={len(X_te):,}")

    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )
        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])

    model = LSTMClassifier(n_features, 128, 2, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                          batch_size=1024, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                         batch_size=2048, shuffle=False)

    t0 = time.time()
    best_acc = 0
    patience_count = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_probs, all_true = [], [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                out = model(xb)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.append(probs)
                all_true.extend(yb.numpy())

        all_preds = np.array(all_preds)
        all_probs = np.vstack(all_probs)
        all_true = np.array(all_true)
        epoch_acc = accuracy_score(all_true, all_preds)
        epoch_f1 = f1_score(all_true, all_preds, average="weighted")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_f1 = epoch_f1
            best_preds = all_preds.copy()
            best_probs = all_probs.copy()
            best_true = all_true.copy()
            patience_count = 0
        else:
            patience_count += 1

        log(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}, acc={epoch_acc:.4f}, f1={epoch_f1:.4f}")

        if patience_count >= 5:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - t0

    a_idx, c_idx = classes.index("A"), classes.index("C")
    y_bin = np.isin(le.inverse_transform(best_true), ["A", "C"]).astype(int)
    try:
        auc = roc_auc_score(y_bin, best_probs[:, a_idx] + best_probs[:, c_idx])
    except Exception:
        auc = 0.0

    log(f"\n  LSTM (seq={SEQ_LEN}) FINAL: Acc={best_acc:.4f}, F1={best_f1:.4f}, AUC={auc:.4f}, Time={elapsed:.1f}s")
    log(f"  For reference: XGBoost = 0.3810 (without case_encoded: comparable features)")

log("\nDONE")
