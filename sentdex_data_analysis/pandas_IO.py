import pandas as pd 

df = pd.read_csv('ZILL-Z77006_C.csv') # reading in file
df.set_index('Date', inplace = True) # setting index to date column

print(df.head())

# df.to_csv('ZILLOW_44106.csv')

df = pd.read_csv('ZILLOW_44106.csv', index_col=0) # reading in file and setting index to the first column

print(df.head())

df.columns = ['Cleveland_HPI'] # House Price Index # renaming the columns

# print(df.head())

# df.to_csv('ZILLOW_44106_Rev3.csv', header = False)

# reading in data, renaming columns, and setting index as first column
df = pd.read_csv('ZILLOW_44106_Rev3.csv', names=['Date', 'Cleveland_HPI'], index_col=0)

# print(df.head())

df.to_html('example.html')  # to HTML (viewable in a web browser)

df = pd.read_csv('ZILLOW_44106_Rev3.csv', names=['Date', 'Cleveland_HPI']) # reading in data and setting headers of columns
print(df.head())

df.rename(columns={'Cleveland_HPI': 'Cleveland_44106_HPI'}, inplace = True) # renaming a column
df.rename(columns={'Cleveland_44106_HPI' : 'Cleveland_HPI'}, inplace=True)
df.set_index('Date', inplace = True)

print(df.head())