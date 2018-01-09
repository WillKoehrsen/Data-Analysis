import pickle
import pandas as pd 
import quandl 
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 
from statistics import mean 
from sklearn import svm
from sklearn.preprocessing import scale, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# housing_pct['ma_apply_example'] = housing_pct['M30'].rolling(window=10).apply(moving_average)
# print(housing_pct.tail())
X = np.array(housing_pct.drop(['label', 'US_HPI_future'], 1))
y = np.array(housing_pct['label'])

X = scale(X)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = svm.SVC(kernel='linear')
# clflog = LogisticRegression(C=50.0, dual=False, penalty="l1")
clflog_accuracy = []
clfsvm_accuracy = []

for i in range(10):
	clflog = LogisticRegression(C=49.0, dual=False, penalty="l1")
	clflog.fit(X_train, y_train)
	clflog_accuracy.append(clflog.score(x_test,y_test))

	clfsvm = svm.SVC(kernel='linear')
	clfsvm.fit(X_train, y_train)
	clfsvm_accuracy.append(clfsvm.score(x_test,y_test))

print('Accuracy of logistic regression = %0.4f' % (mean(clflog_accuracy) * 100))
print('Accuracy of support vector machine = %0.4f' % (mean(clfsvm_accuracy) * 100))