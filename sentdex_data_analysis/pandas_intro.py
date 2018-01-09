import pandas as pd 
import datetime
from pandas_datareader import data
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('seaborn-dark')

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2016, 12, 31)

df = data.DataReader("GM", "yahoo", start, end)

print(df.head())

df['Adj Close'].plot()

plt.show()