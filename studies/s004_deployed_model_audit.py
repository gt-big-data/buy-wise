import sys, os, numpy as np, pandas as pd, joblib
sys.path.insert(0, "/Users/aashishananth/Desktop/CS/buywise/backend")
from ml.dataset import engineer_features, merge_product_features
from ml.inference import _TRAINING_FEATURE_ORDER as FEATS
SP = os.path.dirname(os.path.abspath(__file__))
H = 14

px = pd.read_csv(f"{SP}/Prices.csv", parse_dates=["datetime"])
px.columns = px.columns.str.lower().str.strip()
px = px.rename(columns={"datetime":"date","amazon":"price","new":"new_price",
                        "used":"used_price","listprice":"list_price"})
px["date"] = px["date"].dt.normalize()
px = px[px.price.notna()]

# daily panel (their resample_daily)
out=[]
for a,g in px.sort_values("date").groupby("asin"):
    g=g.groupby("date",as_index=False).last().set_index("date")
    g=g.reindex(pd.date_range(g.index.min(),g.index.max(),freq="D")).ffill(); g["asin"]=a
    out.append(g.reset_index().rename(columns={"index":"date"}))
p = pd.concat(out,ignore_index=True)
p["day_of_week"]=p.date.dt.dayofweek; p["month"]=p.date.dt.month
p["week_of_year"]=p.date.dt.isocalendar().week.astype(int)
p = engineer_features(p)

# Product.csv ships global_sd, pipeline wants global_norm_sd + log_mean_price -> derive
pr = pd.read_csv(f"{SP}/Product.csv"); pr.columns=pr.columns.str.lower().str.strip()
mp = px.groupby("asin").price.mean()
pr["global_norm_sd"]=pr.global_sd/pr.asin.map(mp)
pr["log_mean_price"]=np.log1p(pr.asin.map(mp))
p = merge_product_features(p, pr[["asin","global_min","global_max","global_norm_sd","log_mean_price"]])

# ---- forward truth ----
def fwdmin(s): return s.shift(-1).iloc[::-1].rolling(H,min_periods=1).min().iloc[::-1]
p=p.sort_values(["asin","date"])
p["fwd_min"]=p.groupby("asin")["price"].transform(fwdmin)
p["fwd_last"]=p.groupby("asin")["price"].shift(-H)

# ---- holdout: their 80% date split ----
dates=sorted(p.date.unique()); split=pd.Timestamp(dates[int(len(dates)*0.8)])
print(f"train/test split date = {split.date()}  (models were fit on data before this)")
te=p[(p.date>=split)&p.fwd_min.notna()&p.fwd_last.notna()].copy()

# past-only heuristic features
gb=te.groupby("asin")["price"]
te["min90"]=gb.transform(lambda s:s.shift(1).rolling(90,min_periods=20).min())
te["max90"]=gb.transform(lambda s:s.shift(1).rolling(90,min_periods=20).max())
te=te.dropna(subset=["min90","max90"]).copy()
te["drop_pct"]=(te.price-te.fwd_min)/te.price
te=te[te.drop_pct.between(-1,1)].copy()
print(f"holdout decisions: {len(te):,}  ASINs: {te.asin.nunique()}  base rate P(drop>=8%)={((te.drop_pct>=.08).mean()*100):.1f}%")

# ---- run THEIR saved models ----
scaler=joblib.load("/Users/aashishananth/Desktop/CS/buywise/backend/ml/scaler.joblib")
reg14=joblib.load("/Users/aashishananth/Desktop/CS/buywise/backend/ml/xgb_reg_14d.joblib")
cls14=joblib.load("/Users/aashishananth/Desktop/CS/buywise/backend/ml/xgb_cls_14d.joblib")
X=te.copy()
for f in FEATS:
    if f not in X.columns: X[f]=0.0
sf=list(scaler.feature_names_in_)
X[sf]=X[sf].astype(float).fillna(X[sf].astype(float).median())
X[sf]=scaler.transform(X[sf])
M=X[FEATS].astype(float).fillna(0).values
te["pred_14d"]=reg14.predict(M)
te["proba_wait"]=cls14.predict_proba(M)[:,1]
te["cls"]=cls14.predict(M)

# ---- inference.py decision logic, verbatim ----
exp=(te.price-te.pred_14d)/te.price
te["ml_wait"]=np.where(exp>=0.10,True,np.where(exp<=0.03,False,te.cls==1))

def score(name, wait):
    w=np.asarray(wait,bool)
    catch=np.where(w,te.fwd_min,te.price).mean()
    nocatch=np.where(w,te.fwd_last,te.price).mean()
    y=(te.drop_pct>=0.08).values
    prec=y[w].mean()*100 if w.sum() else np.nan
    return dict(policy=name,wait_pct=w.mean()*100,precision=prec,
                saving_vs_buynow=te.price.mean()-catch, cost_nocatch=nocatch)

rows=[score("always BUY now", np.zeros(len(te),bool)),
      score("ML MODEL (xgb reg+cls)", te.ml_wait.values),
      score("HEURISTIC: top 25% of 90d range", (te.price>(te.min90+0.75*(te.max90-te.min90))).values),
      score("HEURISTIC: price > 90d median", (te.price>te.groupby('asin')['price'].transform(lambda s:s.shift(1).rolling(90,min_periods=20).median())).fillna(False).values),
      score("always WAIT", np.ones(len(te),bool)),
      score("ORACLE", (te.drop_pct>=0.08).values)]
r=pd.DataFrame(rows).set_index("policy")
print("\n=== DOLLARS SAVED PER DECISION vs BUYING NOW (higher = better) ===")
print(r.round(2).to_string())
print(f"\nmean item price in holdout: ${te.price.mean():.2f}")

# regression accuracy
mae=np.abs(te.pred_14d-te.fwd_last).mean()
naive=np.abs(te.price-te.fwd_last).mean()
print(f"\n=== REGRESSION: predicting price at t+14 ===")
print(f"  model MAE:                       ${mae:.2f}")
print(f"  naive 'price stays the same' MAE: ${naive:.2f}")
print(f"  -> model is {'BETTER' if mae<naive else 'WORSE'} than assuming no change by ${abs(mae-naive):.2f}")
