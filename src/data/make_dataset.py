import missingno
import pandas as pd
import numpy as np

# --------------------------------------------------------------
# Load and clean train data
# --------------------------------------------------------------

train = pd.read_csv("../../data/raw/train.csv")
test = pd.read_csv("../../data/raw/test.csv")

combine = pd.concat([train, test], axis = 0).reset_index(drop = True)
combine.head()

combine.info()
combine.isnull().sum().sort_values(ascending = False)
combine.nunique()
combine.describe()
missingno.matrix(combine)

mean_age = combine['Age'].astype('float').mean(axis=0)
combine["Age"].replace(np.nan, mean_age, inplace=True)

mean_fare = combine['Fare'].astype('float').mean(axis=0)
combine["Fare"].replace(np.nan, mean_fare, inplace=True)

is_embarked = combine['Embarked'].value_counts().idxmax()
combine["Embarked"].replace(np.nan, is_embarked, inplace=True)

combine["Cabin"].replace(np.nan,'NC', inplace=True)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

combine.to_csv("../../data/interim/combine_cleaned.csv", index=False)
