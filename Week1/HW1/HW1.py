import pandas as pd
import numpy as np

#1
print(pd.__version__)


df = pd.read_csv("housing.csv")
print(df.head())
#2
print("#2")
print(df.info())
print(df.isnull().values.any())
print(df.total_bedrooms.isnull().sum())

print("#3")
print(df.ocean_proximity.unique())

print("#4")
print(df[df.ocean_proximity == "NEAR BAY"].median_house_value.mean())

print('#5')
print(df.total_bedrooms.mean())

print('#6')
avg = df.total_bedrooms.mean()
df = df.fillna(avg)
print(df.total_bedrooms.fillna(avg).mean())
print(df.isnull().values.any())
print(df.describe())

print('#7')
new_df = df[df.ocean_proximity == "ISLAND"]
X = new_df[['housing_median_age', 'total_rooms', 'total_bedrooms']].values
XTX = X.T.dot(X)
XINV = np.linalg.inv(XTX)
intr = XINV.dot(X.T)
y = np.array([950, 1300, 800, 1000, 1300])
w = intr.dot(y)
print(w[2])
