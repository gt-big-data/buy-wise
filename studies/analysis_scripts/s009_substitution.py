import pandas as pd, numpy as np, os
df=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Prices.csv"),parse_dates=["datetime"])
df.columns=df.columns.str.lower().str.strip()
df=df.rename(columns={"datetime":"date","amazon":"az"})
df=df[df.az.notna()&(df.az>0)]
print(f"raw observations: {len(df):,}  ASINs: {df.asin.nunique()}\n")

# ---- INSTANT SAVING 1: marketplace NEW cheaper than Amazon's price ----
n=df[df.new.notna()&(df.new>0)].copy()
n["gap"]=(n.az-n.new)/n.az
print("=== SWITCH TO CHEAPER *NEW* SELLER (no waiting, same condition) ===")
for t in [0.05,0.10,0.20]:
    m=n.gap>=t
    print(f"  cheaper by >={t*100:>3.0f}%: {m.mean()*100:5.1f}% of observations | avg saving ${(n.az-n.new)[m].mean():6.2f}")
print(f"  unconditional mean saving available: ${np.maximum(n.az-n.new,0).mean():.2f} per observation")

# ---- INSTANT SAVING 2: USED / warehouse cheaper ----
u=df[df.used.notna()&(df.used>0)].copy()
u["gap"]=(u.az-u.used)/u.az
print(f"\n=== BUY *USED*/WAREHOUSE INSTEAD (n={len(u):,}, {len(u)/len(df)*100:.0f}% of obs have a used offer) ===")
for t in [0.10,0.20,0.30]:
    m=u.gap>=t
    print(f"  cheaper by >={t*100:>3.0f}%: {m.mean()*100:5.1f}% of used-available obs | avg saving ${(u.az-u.used)[m].mean():6.2f}")
print(f"  unconditional mean saving available: ${np.maximum(u.az-u.used,0).mean():.2f} per observation (used-available only)")
print(f"  ...spread over ALL observations:     ${np.maximum(u.az-u.used,0).sum()/len(df):.2f}")

# ---- best instant option overall ----
df["best_alt"]=df[["new","used"]].min(axis=1)
b=df[df.best_alt.notna()&(df.best_alt>0)].copy()
b["save"]=np.maximum(b.az-b.best_alt,0)
b["save_pct"]=b.save/b.az
print("\n=== BEST INSTANT ALTERNATIVE (cheapest of new-marketplace / used) ===")
print(f"  mean $ available RIGHT NOW, no waiting, no prediction: ${b.save.mean():.2f}")
print(f"  median: ${b.save.median():.2f} | share of obs with >=$1 available: {(b.save>=1).mean()*100:.1f}%")
for t in [0.05,0.10,0.20]:
    m=b.save_pct>=t
    print(f"  >= {t*100:>3.0f}% cheaper: {m.mean()*100:5.1f}% of obs | avg ${b.save[m].mean():6.2f}")

# ---- LIST PRICE / discount depth sanity ----
l=df[df.listprice.notna()&(df.listprice>0)&(df.listprice>=df.az)].copy()
print(f"\n=== vs LIST PRICE (n={len(l):,}) === already-discounted share: {(l.az<l.listprice).mean()*100:.1f}%")

print("\n=== COMPARISON TO THE TIMING OPPORTUNITY ===")
print(f"  timing (wait up to 14d, perfect execution):  $5.26 per decision")
print(f"  instant substitution (new or used, today):   ${b.save.mean():.2f} per decision")
print(f"  mean Amazon price in sample: ${df.az.mean():.2f}")
