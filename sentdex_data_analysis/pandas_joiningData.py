import pickle
import pandas as pd 
import quandl 
import matplotlib.pyplot as plt 
from matplotlib import style

style.use('seaborn')

quandl.ApiConfig.api_key = 'rFsSehe51RLzREtYhLfo'

def mortgage_30yr():
	df = quandl.get('FMAC/MORTG')
	df = df[df.index > "1974-12-01"]
	df = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100
	df = df.resample('M').mean()
	return df 


ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2, sharex=ax1)

# initial_state_data()

pickle_in = open('fifty_states_pct.pickle' , 'rb')
HPI_data = pickle.load(pickle_in)

# HPI_Benchmark()

pickle_in = open('us_pct.pickle','rb')
benchmark = pickle.load(pickle_in)


m30 = mortgage_30yr()

HPI_Bench = benchmark

state_HPI_M30 = HPI_data.join(m30)
state_HPI_M30.rename({'Value' : 'M30'}, inplace=True)

print(state_HPI_M30.corr().describe()['Value'])