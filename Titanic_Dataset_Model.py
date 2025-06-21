import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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
#  WE DO THIS FOR CATEGORICAL DATA BECAUSE WE CAN KNOW HOW MUCH OF VALUES ARE SHOWN
#sns.countplot(x='Sex', data=df)
#sns.countplot(x = 'Survived', data =df)

#df['Survived'].value_counts().plot(kind='bar')
plt.show()



# LET'S SAY IF WE WANT TO SEE THE INFORMATION IN PERCENTAGE  WE CAN USE PIECHART FOR THIS

df['Pclass'].value_counts().plot(kind='pie', autopct = '%.2f')
plt.show()






# now for the numerical data wala column
# Histogram is the way in which we create ranges for numerical
plt.hist(df['Age'], bins = 100)
plt.show()

# Also we can use distplot
# This is also known as Probability Function
sns.distplot(df['Age'])
plt.show()


# Now we can use BOxplot for noisy data

sns.boxplot(df['Age'])
plt.show()

print(df['Age'].min())
print(df['Age'].max())
print(df['Age'].mean())
print(df['Age'].median())
print(df['Age'].skew())