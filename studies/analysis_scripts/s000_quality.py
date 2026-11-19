import pandas as pd, numpy as np
df = pd.read_csv(f"{__import__('os').path.dirname(__file__)}/Prices.csv", parse_dates=["datetime"])
df.columns = df.columns.str.lower().str.strip()
print("=== RAW ===")
print("rows", len(df), "asins", df.asin.nunique())
print("date span", df.datetime.min().date(), "->", df.datetime.max().date())
print("\nnull rate by col:")
print((df.isna().mean()*100).round(1).to_string())

print("\n=== OBS PER ASIN ===")
n = df.groupby("asin").size()
print(n.describe().round(1).to_string())

print("\n=== GAP BETWEEN CONSECUTIVE OBS (days) ===")
g = df.sort_values(["asin","datetime"]).groupby("asin")["datetime"].diff().dt.days.dropna()
print(g.describe(percentiles=[.1,.25,.5,.75,.9,.99]).round(1).to_string())
print("\ngap value counts (top 15):")
print(g.value_counts().head(15).to_string())
print("\nfrac gaps that are multiples of 7:", round((g[g>0]%7==0).mean(),3))

print("\n=== PER-ASIN DATE SPAN ===")
span = df.groupby("asin")["datetime"].agg(lambda s: (s.max()-s.min()).days)
print(span.describe().round(1).to_string())

print("\n=== AMAZON PRICE ===")
print(df.amazon.describe(percentiles=[.1,.25,.5,.75,.9]).round(2).to_string())
print("rows with amazon null:", df.amazon.isna().sum())

# how many obs per asin AFTER dropping null amazon
d2 = df[df.amazon.notna()]
print("\nafter dropping null amazon: rows", len(d2), "asins", d2.asin.nunique())
print(d2.groupby("asin").size().describe().round(1).to_string())

print("\n=== OBS PER YEAR ===")
print(df.groupby(df.datetime.dt.year).size().to_string())
