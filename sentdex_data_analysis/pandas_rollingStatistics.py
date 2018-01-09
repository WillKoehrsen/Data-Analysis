import pickle
import pandas as pd 
import quandl 
import matplotlib.pyplot as plt 
from matplotlib import style

style.use('seaborn')

api_key = 'rFsSehe51RLzREtYhLfo'

def state_list():
    fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fifty_states[0][0][1:]

def initial_state_data():
    states = state_list()
    main_df = pd.DataFrame()

    for abbv in states:
        query = 'FMAC/HPI_' + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.columns = [str(abbv)]
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fifty_states_pct.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

def HPI_Benchmark():
    df = quandl.get('FMAC/HPI_USA' , authtoken=api_key)
    df['United States'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.0
    
    pickle_out = open('us_pct.pickle', 'wb')
    pickle.dump(df, pickle_out)
    pickle_out.close()

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2, sharex=ax1)

# initial_state_data()

pickle_in = open('fifty_states_pct.pickle' , 'rb')
HPI_data = pickle.load(pickle_in)

# HPI_Benchmark()

pickle_in = open('us_pct.pickle','rb')
benchmark = pickle.load(pickle_in)

# HPI_data = HPI_data.pct_change()

# HPI_data.plot(ax=ax1)
# benchmark['United States'].plot(ax=ax1, color='k', linewidth=10)
# plt.legend().remove()

TX1yr = HPI_data['TX'].resample('A').mean()
HPI_data['TX1yr'] = TX1yr
# print(HPI_data[['TX1yr','TX']])
print(HPI_data.isnull().values.sum())

HPI_data.fillna(method='bfill', inplace=True)
# HPI_data.dropna(inplace=True)
print(HPI_data.isnull().values.sum())

# print(HPI_data[['TX1yr','TX']])

# HPI_data[['TX1yr', 'TX']].plot(ax=ax1)
# plt.show()

# print(HPI_data['TX'].hasnans)

# rolling statistics
HPI_data['TX12MA'] = HPI_data['TX'].rolling(window=12, center=False).mean()
HPI_data['TX12STD']= HPI_data['TX'].rolling(window=12, center=False).std() 
# standard deviation is a measure of the volatility of the price
HPI_data.dropna(inplace=True)

TK_AK_12corr = HPI_data['TX'].rolling(window=12).corr(HPI_data['AK'])

HPI_data['TX'].plot(ax=ax1, label = 'TX HPI')
HPI_data['AK'].plot(ax=ax1, label = 'AK HPI')
ax1.legend(loc=4)

TK_AK_12corr.plot(ax=ax2, label= 'TK AK 12 month correlation')
ax2.legend(loc=4)

# HPI_data[['TX12MA','TX']].plot(ax=ax1)
# HPI_data['TX12STD'].plot(ax=ax2)
# print(HPI_data.head())
plt.show()


