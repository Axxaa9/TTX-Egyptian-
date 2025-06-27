# debug_metadata.py
import pandas as pd

df = pd.read_csv("data/arabic_egy_cleaned/metadata.csv", sep='|', header=None, names=['file', 'text'])
print(df.head())
