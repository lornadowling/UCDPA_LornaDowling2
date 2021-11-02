# import libraries - found a quick fix for issues with import - to look into resolving (during change of methods some libraries may not be needed)
# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
from sklearn.preprocessing import StandardScaler
# noinspection PyUnresolvedReferences
import seaborn as sns
# noinspection PyUnresolvedReferences
from sklearn.preprocessing import StandardScaler
# noinspection PyUnresolvedReferences
from sklearn.preprocessing import LabelEncoder
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import matplotlib.dates as mdates
# noinspection PyUnresolvedReferences
import datetime
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
# noinspection PyUnresolvedReferences
import os #
# noinspection PyUnresolvedReferences
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from sklearn.preprocessing import MinMaxScaler
# noinspection PyUnresolvedReferences
from sklearn.linear_model import LogisticRegression
# noinspection PyUnresolvedReferences
from sklearn.metrics import confusion_matrix
# read csv
BTC_Sales_Data = pd.read_csv("BTC Sales data.csv")
BTC_Sales_info = BTC_Sales_Data.info()
# replace null values with NaN
BTC_Sales = BTC_Sales_Data.replace(0, np.NaN)
print(BTC_Sales.head(10))
# replace NaN with mean
BTC_Sales.fillna(BTC_Sales["Sales"].mean(), inplace=True)
print(BTC_Sales.head(10))
# Plot sales per sector
def sales_per_sector():
    by_sector = BTC_Sales.groupby("Sector")["Sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=by_sector.Sector, y=by_sector.Sales, color='mediumblue')

    ax.set(xlabel = "Sector",
           ylabel = "Sales",
           title = "Sales per Sector")
    plt.show()

    sns.despine()
sales_per_sector()
# Sum sales per month
BTC_Sales = BTC_Sales.groupby("Date")["Sales"].sum().reset_index()
# function to get difference in sales data month on month
def get_diff(BTC_Sales):
    BTC_Sales["sales_diff"] = BTC_Sales.sales.diff()
    BTC_Sales = BTC_Sales.dropna()
    return BTC_Sales
le = LabelEncoder()
for col in BTC_Sales:
        if BTC_Sales[col].dtypes=='object':
                BTC_Sales[col]=le.fit_transform(BTC_Sales[col].astype(str))
for col in BTC_Sales:
        if BTC_Sales[col].dtypes=='float64':
                BTC_Sales[col]=le.fit_transform(BTC_Sales[col].astype(int))
print(BTC_Sales.head(10))
BTC_Sales = BTC_Sales.to_numpy()
X = BTC_Sales[:,0:2]
y = BTC_Sales[:,0]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)
y_pred = logreg.predict(rescaledX_test)
print("Accuracy of logistic regression classifier:", logreg.score(rescaledX_test, y_test))
confusion_matrix(y_test, y_pred)