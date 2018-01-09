import pickle
import pandas as pd 
import quandl 
import matplotlib.pyplot as plt 
from matplotlib import style

style.use('seaborn')

bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
df = pd.DataFrame(bridge_height)

df['std'] = df['meters'].rolling(window=2).std()

df_std = df.describe()['meters']['std']
df_mean = df.describe()['meters']['mean']

# df = df[df['std'] < df_std] # sentdex methods
df = df[df['meters'] < (df_mean + df_std)] # my methods
print(df)

df['meters'].plot()
plt.show()

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2, sharex=ax1)

# initial_state_data()

pickle_in = open('fifty_states_pct.pickle' , 'rb')
HPI_data = pickle.load(pickle_in)

# HPI_Benchmark()

pickle_in = open('us_pct.pickle','rb')
benchmark = pickle.load(pickle_in)

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
# plt.show()


