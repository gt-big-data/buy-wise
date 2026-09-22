import pandas as pd, numpy as np, os
H = 14
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Prices.csv"), parse_dates=["datetime"])
df.columns = df.columns.str.lower().str.strip()
df = df.rename(columns={"datetime":"date","amazon":"price"})
df["date"] = df["date"].dt.normalize()
df = df[df.price.notna()]

# keep asins with enough history
keep = df.groupby("asin")["date"].agg(lambda s:(s.max()-s.min()).days)
keep = keep[keep >= 180].index
df = df[df.asin.isin(keep)]
print(f"ASINs with >=180d span: {df.asin.nunique()} (of 312)")

# daily panel, ffill (replicates their resample_daily)
out=[]
for a,g in df.sort_values("date").groupby("asin"):
    g = g.groupby("date", as_index=False).last().set_index("date")
    idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(idx).ffill(); g["asin"]=a
    out.append(g.reset_index().rename(columns={"index":"date"}))
p = pd.concat(out, ignore_index=True)
print("daily panel rows:", len(p))

# ---- how often does price move at all ----
p = p.sort_values(["asin","date"])
p["chg"] = p.groupby("asin")["price"].pct_change()
print("\n=== DAILY MOVEMENT ===")
print(f"share of days with ANY price change: {(p.chg.abs()>1e-9).mean()*100:.1f}%")
nz = p.chg[p.chg.abs()>1e-9]
print("when it does change, |pct change| percentiles:")
print((nz.abs()*100).describe(percentiles=[.25,.5,.75,.9]).round(2).to_string())

# ---- decision-relevant target: min price over next H days ----
def fwdmin(s): return s.shift(-1).iloc[::-1].rolling(H, min_periods=1).min().iloc[::-1]
p["fwd_min"] = p.groupby("asin")["price"].transform(fwdmin)
p["fwd_last"] = p.groupby("asin")["price"].shift(-H)
p = p[p.fwd_min.notna() & p.fwd_last.notna()].copy()
p["drop_pct"] = (p.price - p.fwd_min)/p.price          # best available saving by waiting
p["drop_usd"] = p.price - p.fwd_min

print(f"\n=== BEST AVAILABLE SAVING IF YOU WAIT UP TO {H} DAYS (n={len(p):,}) ===")
print((p.drop_pct*100).describe(percentiles=[.5,.75,.9,.95,.99]).round(2).to_string())
for t in [0.02,0.05,0.08,0.10,0.15,0.20]:
    print(f"  P(drop >= {t*100:>4.0f}%) = {(p.drop_pct>=t).mean()*100:5.1f}%   mean $ when it happens: ${p.loc[p.drop_pct>=t,'drop_usd'].mean():7.2f}")

print("\n=== ORACLE vs ALWAYS-BUY-NOW (regret, $ per decision) ===")
print(f"  mean saving available (perfect foresight): ${p.drop_usd.mean():.2f}  ({p.drop_pct.mean()*100:.2f}%)")
print(f"  median saving available:                   ${p.drop_usd.median():.2f}")
print(f"  share of decisions where waiting saves $0: {(p.drop_usd<=0.005).mean()*100:.1f}%")

# ---- HOLIDAY HYPOTHESIS ----
print("\n=== IS IT ALL HOLIDAYS? P(drop>=8%) by calendar month ===")
p["month"]=p.date.dt.month
base = (p.drop_pct>=0.08).mean()
m = p.groupby("month").agg(n=("drop_pct","size"), p8=("drop_pct",lambda s:(s>=0.08).mean()),
                           mean_avail=("drop_pct","mean"))
m["lift_vs_base"]=m.p8/base
print((m.assign(p8=(m.p8*100).round(1), mean_avail=(m.mean_avail*100).round(2),
                lift_vs_base=m.lift_vs_base.round(2))).to_string())
print(f"\noverall base rate P(drop>=8%) = {base*100:.1f}%")

# event windows
def in_event(d):
    m,day=d.month,d.day
    if m==11: return "BlackFriday_Nov"
    if m==12 and day<=26: return "Christmas_Dec"
    if m==7: return "PrimeDay_Jul"
    if m==10: return "PrimeBigDeal_Oct"
    return "Normal"
p["window"]=p.date.map(in_event)
print("\n=== EVENT WINDOWS ===")
w=p.groupby("window").agg(n=("drop_pct","size"), p8=("drop_pct",lambda s:(s>=0.08).mean()),
                          mean_pct=("drop_pct","mean"), tot_usd=("drop_usd","sum"))
w["share_of_all_available_savings"]=w.tot_usd/p.drop_usd.sum()
print(w.assign(p8=(w.p8*100).round(1),mean_pct=(w.mean_pct*100).round(2),
               share_of_all_available_savings=(w.share_of_all_available_savings*100).round(1)).to_string())
print(f"\nshare of ROWS that fall in event months: {(p.window!='Normal').mean()*100:.1f}%")
