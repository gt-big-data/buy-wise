"""Where does the clean model work? Segmentation + unseen-ASIN robustness."""
import os,sys,numpy as np,pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"s005_clean_rebuild.py")).read().split('print("\\n=== MATCHED')[0])

Q=0.20; thr=te.proba.quantile(1-Q); te["wait"]=te.proba>=thr
print("\n\n================ SEGMENTATION (model at 20% wait rate) ================")
te["voltile"]=pd.qcut(te.vol30.rank(method="first"),3,labels=["calm","medium","twitchy"])
te["band"]=pd.cut(te.price,[0,25,75,200,1e9],labels=["<$25","$25-75","$75-200",">$200"])
for col in ["voltile","band"]:
    print(f"\n-- by {col} --")
    r=te.groupby(col,observed=True).apply(lambda s:pd.Series({
        "n":len(s),"base_rate_%":s.y.mean()*100,
        "model_precision_%":s.y[s.wait].mean()*100 if s.wait.sum() else np.nan,
        "lift":(s.y[s.wait].mean()/s.y.mean()) if s.wait.sum() and s.y.mean()>0 else np.nan,
        "$_per_decision":s.price.mean()-np.where(s.wait,s.fwd_min,s.price).mean()}),include_groups=False)
    print(r.round(2).to_string())

print("\n\n================ ROBUSTNESS: ASINs NEVER SEEN IN TRAINING ================")
rng=np.random.default_rng(0); asins=p.asin.unique()
held=set(rng.choice(asins,size=int(len(asins)*0.25),replace=False))
tr2=tr[~tr.asin.isin(held)]
te_seen=te[~te.asin.isin(held)]; te_new=te[te.asin.isin(held)]
m2=XGBClassifier(n_estimators=600,learning_rate=0.03,max_depth=5,subsample=.8,colsample_bytree=.8,
                 reg_lambda=2.0,eval_metric="aucpr",early_stopping_rounds=50,n_jobs=-1,random_state=42)
m2.fit(tr2[FEATS],tr2.y,eval_set=[(va[~va.asin.isin(held)][FEATS],va[~va.asin.isin(held)].y)],verbose=False)
for nm,sub in [("ASINs seen in training",te_seen),("ASINs NEVER seen",te_new)]:
    if len(sub)<500: continue
    pr=m2.predict_proba(sub[FEATS])[:,1]; t=np.quantile(pr,1-Q); w=pr>=t
    print(f"{nm:<26} n={len(sub):>6,}  AUC={roc_auc_score(sub.y,pr):.3f}  "
          f"precision@20%={sub.y.values[w].mean()*100:5.1f}%  base={sub.y.mean()*100:4.1f}%  "
          f"lift={sub.y.values[w].mean()/sub.y.mean():.2f}x")

print("\n\n================ CATCH-RATE: does the user actually get the dip? ================")
print("if we alert and they buy at the window low .......... ${:.2f}/decision".format(
      te.price.mean()-np.where(te.wait,te.fwd_min,te.price).mean()))
print("if they only check every 7 days (miss short dips) ... ${:.2f}/decision".format(
      te.price.mean()-np.where(te.wait,te[["price","fwd_last"]].min(axis=1),te.price).mean()))
print("if they just wait 14d and buy whatever ............. ${:.2f}/decision".format(
      te.price.mean()-np.where(te.wait,te.fwd_last,te.price).mean()))
