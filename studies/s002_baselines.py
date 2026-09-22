import pandas as pd, numpy as np, os
H=14
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Prices.csv"), parse_dates=["datetime"])
df.columns=df.columns.str.lower().str.strip()
df=df.rename(columns={"datetime":"date","amazon":"price"})
df["date"]=df["date"].dt.normalize(); df=df[df.price.notna()]
keep=df.groupby("asin")["date"].agg(lambda s:(s.max()-s.min()).days); df=df[df.asin.isin(keep[keep>=180].index)]
out=[]
for a,g in df.sort_values("date").groupby("asin"):
    g=g.groupby("date",as_index=False).last().set_index("date")
    g=g.reindex(pd.date_range(g.index.min(),g.index.max(),freq="D")).ffill(); g["asin"]=a
    out.append(g.reset_index().rename(columns={"index":"date"}))
p=pd.concat(out,ignore_index=True).sort_values(["asin","date"])

# strictly-past features (shift(1) => no lookahead)
gb=p.groupby("asin")["price"]
p["med30"]=gb.transform(lambda s:s.shift(1).rolling(30,min_periods=10).median())
p["med90"]=gb.transform(lambda s:s.shift(1).rolling(90,min_periods=30).median())
p["max90"]=gb.transform(lambda s:s.shift(1).rolling(90,min_periods=30).max())
p["min90"]=gb.transform(lambda s:s.shift(1).rolling(90,min_periods=30).min())
p["vol30"]=gb.transform(lambda s:s.shift(1).pct_change().rolling(30,min_periods=10).std())

def fwdmin(s): return s.shift(-1).iloc[::-1].rolling(H,min_periods=1).min().iloc[::-1]
p["fwd_min"]=p.groupby("asin")["price"].transform(fwdmin)
p["fwd_last"]=p.groupby("asin")["price"].shift(-H)
p=p.dropna(subset=["fwd_min","fwd_last","med30","med90","vol30"]).copy()
p["drop_pct"]=(p.price-p.fwd_min)/p.price
p["y"]=(p.drop_pct>=0.08).astype(int)
p=p[p.drop_pct.between(-1,1)]   # drop absurd outliers
print(f"n = {len(p):,} decisions, {p.asin.nunique()} ASINs,  base rate P(drop>=8%) = {p.y.mean()*100:.1f}%\n")

# ---------- SIGNAL TEST: does a past-only feature separate the classes? ----------
print("=== SIGNAL TEST: P(drop>=8%) by decile of (price / trailing 30d median) ===")
p["rel30"]=p.price/p.med30
p["dec"]=pd.qcut(p.rel30,10,labels=False,duplicates="drop")
s=p.groupby("dec").agg(n=("y","size"),rel_lo=("rel30","min"),rel_hi=("rel30","max"),
                       p8=("y","mean"),mean_avail=("drop_pct","mean"))
print(s.assign(p8=(s.p8*100).round(1),mean_avail=(s.mean_avail*100).round(2),
               rel_lo=s.rel_lo.round(3),rel_hi=s.rel_hi.round(3)).to_string())
print(f"\nspread: bottom decile {s.p8.iloc[0]*100:.1f}%  ->  top decile {s.p8.iloc[-1]*100:.1f}%")

print("\n=== SIGNAL TEST 2: P(drop>=8%) by decile of trailing 30d volatility ===")
p["vdec"]=pd.qcut(p.vol30.rank(method="first"),10,labels=False)
v=p.groupby("vdec").agg(n=("y","size"),vol=("vol30","median"),p8=("y","mean"))
print(v.assign(p8=(v.p8*100).round(1),vol=(v.vol*100).round(2)).to_string())

# ---------- POLICY COMPARISON ----------
print("\n=== POLICY COMPARISON (cost per decision; lower is better) ===")
def evaluate(name, wait_mask):
    w=wait_mask.astype(bool)
    # optimistic: we alert them and they catch the window minimum
    cost_opt=np.where(w,p.fwd_min,p.price)
    # pessimistic: they just wait H days and buy at whatever it is then
    cost_pes=np.where(w,p.fwd_last,p.price)
    prec=p.y[w].mean()*100 if w.sum() else float("nan")
    rec=(p.y[w].sum()/p.y.sum()*100) if p.y.sum() else float("nan")
    return dict(policy=name, wait_pct=w.mean()*100, precision=prec, recall=rec,
                cost_catch=cost_opt.mean(), cost_nocatch=cost_pes.mean())
rows=[evaluate("always BUY now", pd.Series(False,index=p.index)),
      evaluate("always WAIT", pd.Series(True,index=p.index)),
      evaluate("WAIT in November only", p.date.dt.month==11),
      evaluate("WAIT in Nov/Oct/Jul", p.date.dt.month.isin([11,10,7])),
      evaluate("WAIT if price > 30d median", p.price>p.med30),
      evaluate("WAIT if price > 90d median", p.price>p.med90),
      evaluate("WAIT if price >5% over 30d med", p.price>p.med30*1.05),
      evaluate("WAIT if in top 25% of 90d range", p.price>(p.min90+0.75*(p.max90-p.min90))),
      evaluate("ORACLE (perfect foresight)", p.y==1)]
r=pd.DataFrame(rows).set_index("policy")
r["vs_alwaysbuy_catch"]=r.cost_catch-r.loc["always BUY now","cost_catch"]
print(r.round(2).to_string())
print(f"\nmean item price = ${p.price.mean():.2f} | median = ${p.price.median():.2f}")
print("cost_catch   = you WAIT and successfully buy at the window low (requires working alerts)")
print("cost_nocatch = you WAIT and just buy H days later at whatever the price is")
