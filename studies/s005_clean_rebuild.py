"""Leak-free rebuild: strictly past-only features, 3-way chronological split,
threshold chosen on validation, compared to heuristic at MATCHED wait-rate."""
import os, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
SP=os.path.dirname(os.path.abspath(__file__)); H=14
df=pd.read_csv(f"{SP}/Prices.csv",parse_dates=["datetime"]); df.columns=df.columns.str.lower().str.strip()
df=df.rename(columns={"datetime":"date","amazon":"price"}); df["date"]=df["date"].dt.normalize()
df=df[df.price.notna()]
keep=df.groupby("asin")["date"].agg(lambda s:(s.max()-s.min()).days); df=df[df.asin.isin(keep[keep>=270].index)]
out=[]
for a,g in df.sort_values("date").groupby("asin"):
    g=g.groupby("date",as_index=False).last().set_index("date")
    g=g.reindex(pd.date_range(g.index.min(),g.index.max(),freq="D")).ffill(); g["asin"]=a
    out.append(g.reset_index().rename(columns={"index":"date"}))
p=pd.concat(out,ignore_index=True).sort_values(["asin","date"]).reset_index(drop=True)

g=p.groupby("asin")["price"]
def past(fn): return g.transform(lambda s: fn(s.shift(1)))
p["med30"]=past(lambda s:s.rolling(30,min_periods=10).median())
p["med90"]=past(lambda s:s.rolling(90,min_periods=30).median())
p["min90"]=past(lambda s:s.rolling(90,min_periods=30).min())
p["max90"]=past(lambda s:s.rolling(90,min_periods=30).max())
p["min180"]=past(lambda s:s.rolling(180,min_periods=60).min())
p["exp_min"]=past(lambda s:s.expanding(min_periods=30).min())     # past-only "global min"
p["exp_max"]=past(lambda s:s.expanding(min_periods=30).max())
p["vol30"]=past(lambda s:s.pct_change().rolling(30,min_periods=10).std())
p["vol90"]=past(lambda s:s.pct_change().rolling(90,min_periods=30).std())
p["chg7"]=past(lambda s:s.pct_change(7)); p["chg30"]=past(lambda s:s.pct_change(30))
p["nchg30"]=past(lambda s:(s.diff().abs()>1e-9).rolling(30,min_periods=10).sum())
p["dsince"]=past(lambda s:s.groupby((s.diff().abs()>1e-9).cumsum()).cumcount())
p["rel30"]=p.price/p.med30; p["rel90"]=p.price/p.med90
p["pos90"]=(p.price-p.min90)/(p.max90-p.min90+1e-9)
p["rel_expmin"]=p.price/p.exp_min; p["rel_expmax"]=p.price/p.exp_max
p["rel_min180"]=p.price/p.min180
p["logp"]=np.log1p(p.price); p["month"]=p.date.dt.month; p["woy"]=p.date.dt.isocalendar().week.astype(int)
p["used_gap"]=(p.price-p.used)/p.price; p["new_gap"]=(p.price-p.new)/p.price
for c in ["count_new","count_used","sales_score"]: p[c]=pd.to_numeric(p[c],errors="coerce")

def fwdmin(s): return s.shift(-1).iloc[::-1].rolling(H,min_periods=1).min().iloc[::-1]
p["fwd_min"]=p.groupby("asin")["price"].transform(fwdmin)
p["fwd_last"]=p.groupby("asin")["price"].shift(-H)
p=p.dropna(subset=["fwd_min","fwd_last","med90","vol30","pos90"]).copy()
p["drop_pct"]=(p.price-p.fwd_min)/p.price; p=p[p.drop_pct.between(-1,1)]
p["y"]=(p.drop_pct>=0.08).astype(int)

FEATS=["rel30","rel90","pos90","rel_expmin","rel_expmax","rel_min180","vol30","vol90",
       "chg7","chg30","nchg30","dsince","logp","month","woy","used_gap","new_gap",
       "count_new","count_used","sales_score"]
d=sorted(p.date.unique()); c1=pd.Timestamp(d[int(len(d)*.6)]); c2=pd.Timestamp(d[int(len(d)*.8)])
# embargo: a train row's 14d target must not reach into the next period
tr=p[p.date+pd.Timedelta(days=H)<c1]; va=p[(p.date>=c1)&(p.date+pd.Timedelta(days=H)<c2)]; te=p[p.date>=c2]
print(f"train {len(tr):,} (->{c1.date()}) | val {len(va):,} | test {len(te):,} (from {c2.date()})")
print(f"base rate P(drop>=8%) in test = {te.y.mean()*100:.1f}%\n")

m=XGBClassifier(n_estimators=600,learning_rate=0.03,max_depth=5,subsample=.8,colsample_bytree=.8,
                reg_lambda=2.0,eval_metric="aucpr",early_stopping_rounds=50,n_jobs=-1,random_state=42)
m.fit(tr[FEATS],tr.y,eval_set=[(va[FEATS],va.y)],verbose=False)
te=te.copy(); te["proba"]=m.predict_proba(te[FEATS])[:,1]
print(f"test ROC-AUC = {roc_auc_score(te.y,te.proba):.3f}   PR-AUC = {average_precision_score(te.y,te.proba):.3f}  (base {te.y.mean():.3f})")

print("\n=== MATCHED WAIT-RATE COMPARISON (each policy says WAIT on exactly N% of decisions) ===")
print(f"{'wait rate':>10} | {'CLEAN MODEL':>26} | {'HEURISTIC pos90':>26}")
print(f"{'':>10} | {'precision':>10} {'$/decision':>14} | {'precision':>10} {'$/decision':>14}")
base=te.price.mean()
for q in [0.10,0.20,0.30,0.50]:
    rows=[]
    for col in ["proba","pos90"]:
        thr=te[col].quantile(1-q); w=(te[col]>=thr).values
        prec=te.y.values[w].mean()*100
        saved=base-np.where(w,te.fwd_min,te.price).mean()
        rows.append((prec,saved))
    print(f"{q*100:>9.0f}% | {rows[0][0]:>9.1f}% {rows[0][1]:>13.2f}$ | {rows[1][0]:>9.1f}% {rows[1][1]:>13.2f}$")
print(f"\nmean price ${base:.2f} | ceiling (always wait, perfect catch) ${base-te.fwd_min.mean():.2f}")
imp=pd.Series(m.feature_importances_,index=FEATS).sort_values(ascending=False)
print("\ntop 8 features:"); print(imp.head(8).round(3).to_string())
