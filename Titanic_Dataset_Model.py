import pandas as pd


df = pd.read_csv("D:/New Shop Prediction Model/Titanic-Dataset.csv")
print(df.head())
print("\n\n\n")

pd.set_option('display.max_columns', None)
print(df.sample(10))
