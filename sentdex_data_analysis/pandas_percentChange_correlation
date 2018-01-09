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

	print(main_df.head())

	pickle_out = open('fifty_states_pct.pickle', 'wb')
	pickle.dump(main_df, pickle_out)
	pickle_out.close()



def HPI_Benchmark():
	df = quandl.get('FMAC/HPI_USA' , authtoken=api_key)
	df['United States'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.0
	
	pickle_out = open('us_pct.pickle', 'wb')
	pickle.dump(df, pickle_out)
	pickle_out.close()

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

# initial_state_data()
pickle_in = open('fifty_states_pct.pickle' , 'rb')
HPI_data = pickle.load(pickle_in)


# HPI_Benchmark()
pickle_in = open('us_pct.pickle' , 'rb')
benchmark = pickle.load(pickle_in)

# HPI_data = HPI_data.pct_change()

HPI_data.plot(ax=ax1)
benchmark['United States'].plot(ax=ax1, color='k', linewidth=10)
plt.legend().remove()

HPI_State_Correlation = HPI_data.corr()
print(HPI_State_Correlation)

plt.show()
# print(HPI_data[['IL','WI']].corr())
