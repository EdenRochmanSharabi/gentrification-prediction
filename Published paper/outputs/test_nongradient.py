"""
Test non-gradient-boosting algorithms for gentrification prediction.
Focus on fundamentally different approaches:
1. MLP (neural network)
2. LSTM (recurrent — captures temporal sequence per CUSEC)
3. SVM (kernel methods)
4. Logistic Regression (linear baseline)
5. KNN
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter

from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "nongradient.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("NON-GRADIENT ALGORITHM COMPARISON")
log("=" * 60)

# ── Load data (same as model_leakfree.py) ──
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

# Features
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
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "sale_change_curr", "rent_change_curr",
    "sale_change_lag1", "rent_change_lag1",
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6",
    "sale_momentum", "rent_momentum", "change_interaction",
    "sale_vs_prov", "rent_vs_prov",
    "sale_yoy", "rent_yoy",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "month", "quarter", "year",
]

log(f"Features: {len(FEATURES)}")

trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])

train_mask = trainable["year"] < 2020
X_train = trainable.loc[train_mask, FEATURES].values
X_test = trainable.loc[~train_mask, FEATURES].values
y_train = trainable.loc[train_mask, "target_encoded"].values
y_test = trainable.loc[~train_mask, "target_encoded"].values

log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
log(f"Classes: {list(le.classes_)}")

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_train_p = sc.fit_transform(imp.fit_transform(X_train))
X_test_p = sc.transform(imp.transform(X_test))

classes = list(le.classes_)
y_bin_test = np.isin(trainable.loc[~train_mask, "target_case"], ["A", "C"]).astype(int)
a_idx, c_idx = classes.index("A"), classes.index("C")

results = []

def evaluate(name, clf, X_tr, X_te, y_tr, y_te, has_proba=True):
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="weighted")
    if has_proba:
        try:
            proba = clf.predict_proba(X_te)
            auc = roc_auc_score(y_bin_test, proba[:, a_idx] + proba[:, c_idx])
        except Exception:
            auc = None
    else:
        auc = None
    auc_str = f"{auc:.4f}" if auc else "N/A"
    log(f"  Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_str}, Time={elapsed:.1f}s")
    results.append({"model": name, "acc": acc, "f1": f1, "auc": auc, "time": elapsed})

# ── 1. Logistic Regression ──
log("\n" + "=" * 60)
log("MODEL: Logistic Regression")
evaluate("Logistic Regression",
         LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", C=1.0, random_state=42),
         X_train_p, X_test_p, y_train, y_test)

# ── 2. Linear SVM ──
log("\n" + "=" * 60)
log("MODEL: Linear SVM")
evaluate("Linear SVM",
         LinearSVC(max_iter=2000, random_state=42, dual=True),
         X_train_p, X_test_p, y_train, y_test, has_proba=False)

# ── 3. KNN ──
log("\n" + "=" * 60)
log("MODEL: KNN (k=15)")
evaluate("KNN (k=15)",
         KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
         X_train_p, X_test_p, y_train, y_test)

# ── 4. MLP (small) ──
log("\n" + "=" * 60)
log("MODEL: MLP (128-64)")
evaluate("MLP (128-64)",
         MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True,
                       validation_fraction=0.1, random_state=42, learning_rate="adaptive"),
         X_train_p, X_test_p, y_train, y_test)

# ── 5. MLP (large) ──
log("\n" + "=" * 60)
log("MODEL: MLP (256-128-64)")
evaluate("MLP (256-128-64)",
         MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, early_stopping=True,
                       validation_fraction=0.1, random_state=42, learning_rate="adaptive"),
         X_train_p, X_test_p, y_train, y_test)

# ── 6. LSTM (PyTorch) ──
log("\n" + "=" * 60)
log("MODEL: LSTM (sequence of 4 quarters per sample)")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    SEQ_LEN = 4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"  Device: {device}")

    # Build sequences: for each row, use the previous SEQ_LEN rows of the same CUSEC
    log("  Building sequences...")
    trainable_sorted = trainable.sort_values(["CUSEC", "period"])

    def build_sequences(data, mask, features, seq_len):
        X_seq, y_seq = [], []
        cusec_groups = data[mask].groupby("CUSEC")
        for cusec, group in cusec_groups:
            vals = group[features].values
            targets = group["target_encoded"].values
            for i in range(seq_len, len(vals)):
                X_seq.append(vals[i-seq_len:i])
                y_seq.append(targets[i])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

    imp_seq = SimpleImputer(strategy="mean")
    sc_seq = StandardScaler()
    trainable_sorted[FEATURES] = imp_seq.fit_transform(trainable_sorted[FEATURES])
    trainable_sorted[FEATURES] = sc_seq.fit_transform(trainable_sorted[FEATURES])

    X_seq_train, y_seq_train = build_sequences(trainable_sorted, train_mask.values, FEATURES, SEQ_LEN)
    X_seq_test, y_seq_test = build_sequences(trainable_sorted, ~train_mask.values, FEATURES, SEQ_LEN)

    log(f"  Sequences: train={len(X_seq_train):,}, test={len(X_seq_test):,}")

    n_features = len(FEATURES)
    n_classes = len(classes)

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

    train_ds = TensorDataset(torch.from_numpy(X_seq_train), torch.from_numpy(y_seq_train))
    test_ds = TensorDataset(torch.from_numpy(X_seq_test), torch.from_numpy(y_seq_test))
    train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=2048, shuffle=False)

    t0 = time.time()
    best_acc = 0
    patience = 3
    no_improve = 0

    for epoch in range(20):
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

        # Eval
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

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_preds = all_preds.copy()
            best_probs = all_probs.copy()
            best_true = all_true.copy()
            no_improve = 0
        else:
            no_improve += 1

        log(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}, test_acc={epoch_acc:.4f}")

        if no_improve >= patience:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    acc = accuracy_score(best_true, best_preds)
    f1 = f1_score(best_true, best_preds, average="weighted")

    y_bin_lstm = np.isin(le.inverse_transform(best_true), ["A", "C"]).astype(int)
    auc = roc_auc_score(y_bin_lstm, best_probs[:, a_idx] + best_probs[:, c_idx])

    log(f"  LSTM FINAL: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Time={elapsed:.1f}s")
    results.append({"model": "LSTM (seq=4)", "acc": acc, "f1": f1, "auc": auc, "time": elapsed})

except ImportError:
    log("  PyTorch not available, skipping LSTM")
except Exception as e:
    log(f"  LSTM error: {e}")
    import traceback
    traceback.print_exc()

# ── Summary ──
log("\n" + "=" * 60)
log("SUMMARY (Non-gradient algorithms)")
log("=" * 60)
log(f"{'Model':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Time':>8}")
log("-" * 60)
results.sort(key=lambda x: x["acc"], reverse=True)
for r in results:
    auc_str = f"{r['auc']:.4f}" if r["auc"] else "N/A"
    log(f"{r['model']:<25} {r['acc']:>8.4f} {r['f1']:>8.4f} {auc_str:>8} {r['time']:>7.1f}s")
log("")
log(f"BEST: {results[0]['model']} (Acc={results[0]['acc']:.4f})")
log("For reference: XGBoost = 0.3810 / Gradient boosters ~ 0.381")
log("DONE")
