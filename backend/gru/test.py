"""
Tests a model based on data contained in data/test_7day.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import time

# ----------------------------
# settings
# ----------------------------
seq_len = 30
batch_size = 256

model_path = "gru_model.pt"
scaler_path = "scaler.joblib"
features_path = "features.joblib"
asin_map_path = "asin_map.joblib"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# load data
# ----------------------------
print("loading test data...")
t0 = time.time()

test = pd.read_csv("data/test_7day.csv")

print(f"done in {time.time() - t0:.2f}s")

features = joblib.load(features_path)
asin_map = joblib.load(asin_map_path)
scaler = joblib.load(scaler_path)

print("features:", len(features))
print("asins:", len(asin_map))


# map product ids
test["asin_id"] = test["asin"].map(asin_map).fillna(0).astype(int)


# ----------------------------
# targets (only for evaluation)
# ----------------------------
test["target_7d"] = np.log1p(test["label_7d"]) - np.log1p(test["price"])
test["target_14d"] = np.log1p(test["label_14d"]) - np.log1p(test["price"])

targets = ["target_7d", "target_14d"]


# ----------------------------
# clean + scale features
# ----------------------------
test[features] = test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
test[features] = scaler.transform(test[features])


# ----------------------------
# turn data into short sequences
# ----------------------------
def create_sequences(df):
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
            A.append(a_vals[i + seq_len - 1])

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(Y), dtype=torch.float32),
        torch.tensor(np.array(A), dtype=torch.long),
    )


print("building sequences...")
X, y, A = create_sequences(test)

print("shape:", X.shape)


# ----------------------------
# model
# ----------------------------
class AttnPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):
        weights = self.attn(x).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


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
        emb = self.emb(a).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, emb], dim=-1)

        out, _ = self.gru(x)
        out = self.pool(out)

        return self.head(out)


print("loading model...")
checkpoint = torch.load(model_path, map_location=device)

model = Model(len(features), len(asin_map)).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()


# ----------------------------
# run predictions
# ----------------------------
print("running inference...")

preds = []

with torch.no_grad():
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size].to(device)
        ab = A[i:i+batch_size].to(device)

        preds.append(model(xb, ab).cpu().numpy())

preds = np.vstack(preds)
true = y.numpy()


# ----------------------------
# results
# ----------------------------
print("\nresults")
print("7d mse:", np.mean((preds[:, 0] - true[:, 0]) ** 2))
print("14d mse:", np.mean((preds[:, 1] - true[:, 1]) ** 2))
print("done")