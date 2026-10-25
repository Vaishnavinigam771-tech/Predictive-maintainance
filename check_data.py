import pandas as pd

df = pd.read_csv("data/ai4i2020.csv")

print(df.shape)
print(df.columns.tolist())