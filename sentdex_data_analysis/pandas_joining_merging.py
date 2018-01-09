import pandas as pd

df1 = pd.DataFrame({'HPI':[80,86,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55],
                   'Year' : [2001, 2002, 2003, 2005]})

'''
df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[5, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2005, 2006, 2007, 2008])
'''

df3 = pd.DataFrame({'HPI':[95, 86, 88, 90],
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53],
                   'Year' : [2000, 2002, 2003, 2004]})


# print(pd.merge(df1, df3, on=['HPI']))
# print(df1)
# print(df3)

# df1.set_index('Year', inplace=True)
# df3.set_index('Year', inplace=True)



merged = pd.merge(df1, df3, on='Year', how='outer')
print(merged)