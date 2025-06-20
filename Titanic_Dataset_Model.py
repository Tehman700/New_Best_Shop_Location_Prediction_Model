import pandas as pd
import seaborn as sns

df = pd.read_csv("D:/New Shop Prediction Model/Titanic-Dataset.csv")
print(df.head())
print("\n\n\n")

# THIS IS USED FOR DISPLAYING ALL THE COLUMNS
pd.set_option('display.max_columns', None)
print(df.sample(10))

# BELOW IS USED FOR THE FINDINGS OF DATATYPE OF COLUMNS
print(df.info())

# ARE THERE ANY MISSING VALUES IN THE DATASET
print(df.isnull().sum())

# HOW THE DATASET LOOKS MATHEMATICALLY
print(df.describe())

# H OIW THE FINDINGS DUPLICATE VALUES
print(df.duplicated().sum())
print("\n\n\n\n\n\n\n")

# THE ABOVE IS THE FORMAL STEPS APPLIED FOR A DATASET TO GET THE KNOW HOW OF THE DATASET

# BELOW IS THE APPLYING OF EDA UNIVARIATE ANALYSIS
# HAR COLUMN PR INDIVIDUALLY KAM KRTAY
# BY USING GRAPHS

print(df.head(10))


# let's work on survived column as graphing