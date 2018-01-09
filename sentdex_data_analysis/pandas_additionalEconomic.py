import pickle
import pandas as pd 
import quandl 
import matplotlib.pyplot as plt 
from matplotlib import style

style.use('seaborn')

quandl.ApiConfig.api_key = 'rFsSehe51RLzREtYhLfo'

def mortgage_30yr():
	df = quandl.get('FMAC/MORTG', trim_start="1975-01-01")
	df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100
	df = df.resample('M').mean()
	df.rename(columns={'Value': 'M30'}, inplace=True)
	df = df['M30']
	return df 

def sp500_data():
    df = quandl.get("YAHOO/INDEX_GSPC", trim_start="1975-01-01")
    df["Adjusted Close"] = (df["Adjusted Close"]-df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Adjusted Close':'sp500'}, inplace=True)
    df = df['sp500']
    return df

def gdp_data():
    df = quandl.get("BCB/4385", trim_start="1975-01-01")
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df = df['GDP'] # DataFrame to Series
    return df

def us_unemployment():
    df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01")
    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    return df

# m30 = mortgage_30yr() # Series
# sp500 = sp500_data() # Series
# gdp = gdp_data() # Series
# unemployment = us_unemployment() # DataFrame
# HPI = HPI_data.join([m30, unemployment, gdp, sp500])

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2, sharex=ax1)

# initial_state_data()

pickle_in = open('fifty_states_pct.pickle' , 'rb')
HPI_data = pickle.load(pickle_in)

# HPI_Benchmark()

pickle_in = open('us_pct.pickle','rb')
benchmark = pickle.load(pickle_in)

pickle_in = open('HPI_complete.pickle', 'rb')
HPI = pickle.load(pickle_in)
HPI.dropna(inplace=True)
print(HPI.head())


state_HPI_M30 = HPI_data.join(HPI['M30'])


# print(state_HPI_M30.corr().describe()['M30'])