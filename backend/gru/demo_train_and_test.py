"""
Runs minimal training & testing on a subset of the data, for testing purposes.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import time


# ----------------------------
# CONFIG
# ----------------------------
seq_len = 30
batch_size = 128
epochs = 5
lr = 1e-3

MAX_ASINS = 100
MAX_ROWS_PER_ASIN = 1000
STRIDE = 3

model_path = "demo_gru.pt"
scaler_path = "demo_scaler.joblib"
features_path = "demo_features.joblib"
asin_map_path = "demo_asin_map.joblib"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# dataset wrapper
# ----------------------------
class DS(Dataset):
    def __init__(self, X, y, A):
        self.X = X
        self.y = y
        self.A = A

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.A[i]


# ----------------------------
# model
# ----------------------------
class Model(nn.Module):
    def __init__(self, input_size, n_asins, emb_dim=16, hidden=64):
        super().__init__()

        self.emb = nn.Embedding(n_asins + 1, emb_dim)

        self.gru = nn.GRU(
            input_size=input_size + emb_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, a):
        e = self.emb(a).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, e], dim=-1)

        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# ----------------------------
# sequence builder
# ----------------------------
def create_sequences(df, features, targets):
    X, Y, A = [], [], []

    for asin, g in df.groupby("asin"):
        if len(g) <= seq_len:
            continue

        g = g.sort_values("keepa_minutes")

        x_vals = g[features].values.astype(np.float32)
        y_vals = g[targets].values.astype(np.float32)
        a_vals = g["asin_id"].values.astype(np.int64)

        for i in range(0, len(g) - seq_len, STRIDE):
            yv = y_vals[i + seq_len]

            if np.isnan(yv).any():
                continue

            X.append(x_vals[i:i+seq_len])
            Y.append(yv)
            A.append(a_vals[i + seq_len - 1])  # ✅ fixed alignment

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(Y), dtype=torch.float32),
        torch.tensor(np.array(A), dtype=torch.long),
    )


# ----------------------------
# main
# ----------------------------
def main():
    print("loading data...")
    t0 = time.time()

    df = pd.read_csv("data/train_7day.csv")

    # limit ASINs
    top_asins = df["asin"].value_counts().head(MAX_ASINS).index
    df = df[df["asin"].isin(top_asins)]

    # cap rows
    df = df.sort_values(["asin", "keepa_minutes"])
    df = df.groupby("asin").head(MAX_ROWS_PER_ASIN)

    print(f"loaded in {time.time()-t0:.2f}s | rows={len(df)} | asins={df['asin'].nunique()}")


    # ----------------------------
    # targets
    # ----------------------------
    df["target_7d"] = np.log1p(df["label_7d"]) - np.log1p(df["price"])
    df["target_14d"] = np.log1p(df["label_14d"]) - np.log1p(df["price"])
    targets = ["target_7d", "target_14d"]


    # ----------------------------
    # features
    # ----------------------------
    drop_cols = ["date", "label_7d", "label_14d", "target_7d", "target_14d"]

    features = [
        c for c in df.columns
        if c not in drop_cols + ["asin", "brand"]
    ]

    joblib.dump(features, features_path)
    print("features:", len(features))


    # ----------------------------
    # asin encoding
    # ----------------------------
    asin_map = {a: i + 1 for i, a in enumerate(df["asin"].unique())}
    df["asin_id"] = df["asin"].map(asin_map).astype(int)

    joblib.dump(asin_map, asin_map_path)


    # ----------------------------
    # clean + scale
    # ----------------------------
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    df[targets] = df[targets].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    joblib.dump(scaler, scaler_path)


    # ----------------------------
    # build sequences
    # ----------------------------
    print("building sequences...")
    X, y, A = create_sequences(df, features, targets)

    print("dataset shape:", X.shape)


    # ----------------------------
    # train / test split (time-safe)
    # ----------------------------
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    A_train, A_test = A[:split], A[split:]


    # ----------------------------
    # loader
    # ----------------------------
    loader = DataLoader(
        DS(X_train, y_train, A_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )


    # ----------------------------
    # model
    # ----------------------------
    model = Model(len(features), len(asin_map)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("training...")


    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb, ab in loader:
            xb, yb, ab = xb.to(device), yb.to(device), ab.to(device)

            opt.zero_grad()
            pred = model(xb, ab)

            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"epoch {epoch+1} done | loss {total_loss:.4f}")


    # ----------------------------
    # evaluation
    # ----------------------------
    print("evaluating...")
    model.eval()

    preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i:i+batch_size].to(device)
            ab = A_test[i:i+batch_size].to(device)

            preds.append(model(xb, ab).cpu().numpy())

    preds = np.vstack(preds)
    true = y_test.numpy()


    print("\nresults")
    print("mse:", np.mean((preds - true) ** 2))
    print("7d mse:", np.mean((preds[:, 0] - true[:, 0]) ** 2))
    print("14d mse:", np.mean((preds[:, 1] - true[:, 1]) ** 2))


    # ----------------------------
    # save
    # ----------------------------
    torch.save(
        {
            "model": model.state_dict(),
            "input_size": len(features)
        },
        model_path
    )

    print("saved model")


if __name__ == "__main__":
    main()