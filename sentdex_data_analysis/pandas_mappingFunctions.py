import pickle
import pandas as pd 
import quandl 
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 
from statistics import mean 

style.use('seaborn-dark-palette')

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2, sharex=ax1)

def create_labels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1
    else:
        return 0

def moving_average(values):
    return mean(values)

benchmark = pd.read_pickle('us_pct.pickle')  # us overall housing price index percentage change
HPI = pd.read_pickle('HPI_complete.pickle') # all of the state data, thirty year mortgage, unemployment rate, GDP, SP500
HPI = HPI.join(benchmark['United States'])
# all in percentage change since the start of the data (1975-01-01)

HPI.dropna(inplace=True)

housing_pct = HPI.pct_change()
housing_pct.replace([np.inf, -np.inf], np.nan, inplace=True)

housing_pct['US_HPI_future'] = housing_pct['United States'].shift(-1)
housing_pct.dropna(inplace=True)

housing_pct['label'] = list(map(create_labels, housing_pct['United States'], housing_pct['US_HPI_future']))

# housing_pct['ma_apply_example'] = pd.rolling_apply(housing_pct['M30'], 10, moving_average)
housing_pct['ma_apply_example'] = housing_pct['M30'].rolling(window=10).apply(moving_average)
print(housing_pct.tail())

# state_HPI_M30 = HPI_data.join(HPI['M30']) # fifty states plus mortgage data
# print(state_HPI_M30.corr().describe().tail())