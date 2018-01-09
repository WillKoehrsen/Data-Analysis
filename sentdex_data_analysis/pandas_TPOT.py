import pandas as pd 
import numpy as np 
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

benchmark = pd.read_pickle('us_pct.pickle')  # us overall housing price index percentage change
HPI = pd.read_pickle('HPI_complete.pickle') # all of the state data, thirty year mortgage, unemployment rate, GDP, SP500
HPI = HPI.join(benchmark['United States'])
# all in percentage change since the start of the data (1975-01-01)

HPI.dropna(inplace=True)

housing_pct = HPI.pct_change()
housing_pct.replace([np.inf, -np.inf], np.nan, inplace=True)

housing_pct['US_HPI_future'] = housing_pct['United States'].shift(-1)
housing_pct.dropna(inplace=True)

def create_labels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1
    else:
        return 0

housing_pct['label'] = list(map(create_labels, housing_pct['United States'], housing_pct['US_HPI_future']))

# housing_pct['ma_apply_example'] = housing_pct['M30'].rolling(window=10).apply(moving_average)
# print(housing_pct.tail())
X = np.array(housing_pct.drop(['label', 'US_HPI_future'], 1))
y = np.array(housing_pct['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('HPI_tpot_pipeline.py')