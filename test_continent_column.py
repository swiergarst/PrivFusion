import pandas as pd

dataset = pd.read_csv("datasets/covid_global.csv")

col = "continent"
s = dataset[col]

print(s.dtype)
print(s.isna().sum(), "nulls of", len(s))
print(s.nunique(), "unique values")
print(s.map(type).value_counts())  # mixed types?
print(s.value_counts(dropna=False).head(10))
print(s.astype(str).str.len().describe())  # string lengths
