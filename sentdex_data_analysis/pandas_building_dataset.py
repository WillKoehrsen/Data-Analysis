import quandl
import pandas as pd 

api_key = 'rFsSehe51RLzREtYhLfo'

# df = quandl.get('FMAC/HPI_AK', authtoken = api_key)

fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')

for abbv in fifty_states[0][0][1:]:
	print('FMAC/HPI_' + str(abbv))


