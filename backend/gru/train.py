"""
Trains a model based on data contained in data/train_7day.csv
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
# basic settings
# ----------------------------
seq_len = 30
batch_size = 256
epochs = 10
lr = 2e-4

model_path = "gru_model.pt"
scaler_path = "scaler.joblib"
features_path = "features.joblib"
asin_map_path = "asin_map.joblib"

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
# attention pooling (small helper)
# ----------------------------
class AttnPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, x):
        w = self.score(x).squeeze(-1)
        w = torch.softmax(w, dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


# ----------------------------
# model
# ----------------------------
class Model(nn.Module):
    def __init__(self, input_size, n_asins, emb_dim=24, hidden=128):
        super().__init__()

        self.emb = nn.Embedding(n_asins + 1, emb_dim)

        self.gru = nn.GRU(
            input_size=input_size + emb_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        self.pool = AttnPool(hidden * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x, a):
        e = self.emb(a).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, e], dim=-1)

        out, _ = self.gru(x)
        out = self.pool(out)

        return self.head(out)


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

        for i in range(len(g) - seq_len):
            yv = y_vals[i + seq_len]

            if np.isnan(yv).any():
                continue

            X.append(x_vals[i:i+seq_len])
            Y.append(yv)
            A.append(a_vals[i])

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

    train = pd.read_csv("data/train_7day.csv")

    # small cap for testing (remove for full run)
    train = train.sort_values(["asin", "keepa_minutes"])
    train = train.groupby("asin").head(100)

    print(f"loaded in {time.time() - t0:.2f}s")


    # targets
    train["target_7d"] = np.log1p(train["label_7d"]) - np.log1p(train["price"])
    train["target_14d"] = np.log1p(train["label_14d"]) - np.log1p(train["price"])
    targets = ["target_7d", "target_14d"]


    # features
    drop_cols = ["date", "label_7d", "label_14d", "target_7d", "target_14d"]

    features = [
        c for c in train.columns
        if c not in drop_cols + ["asin", "brand"] # string-based entries
    ]

    joblib.dump(features, features_path)
    print("features:", len(features))


    # asin encoding
    asin_map = {a: i + 1 for i, a in enumerate(train["asin"].unique())}
    train["asin_id"] = train["asin"].map(asin_map).astype(int)

    joblib.dump(asin_map, asin_map_path)


    # clean + scale
    train[features] = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])

    joblib.dump(scaler, scaler_path)


    # build sequences
    print("building sequences...")
    X, y, A = create_sequences(train, features, targets)

    print("dataset shape:", X.shape)


    # loader
    loader = DataLoader(
        DS(X, y, A),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=2  # try 0 if not working on windows
    )


    # model
    model = Model(len(features), len(asin_map)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()

    print("training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (xb, yb, ab) in enumerate(loader):
            xb, yb, ab = xb.to(device), yb.to(device), ab.to(device)

            opt.zero_grad()
            pred = model(xb, ab)

            loss = loss_fn(pred, yb)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * xb.size(0)

        print(f"epoch {epoch} done, avg loss {total_loss / len(loader.dataset):.4f}")


    # save
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