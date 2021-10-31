import pandas as pd
# noinspection PyUnresolvedReferences
import numpy as np
BTC_Sales_Data = pd.read_csv("BTC Sales data.csv")
BTC_Sales_info = BTC_Sales_Data.info()
BTC_Sales = BTC_Sales_Data.replace(0, np.NaN)
print(BTC_Sales.head(10))