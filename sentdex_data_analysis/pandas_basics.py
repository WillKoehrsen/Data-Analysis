import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')

web_stats = {'Day': [1,2,3,4,5,6],
			 'Visitors': [54, 65, 76, 76, 34, 34],
			 'Bounce_Rate': [54, 23, 32, 54, 54, 32]}

df = pd.DataFrame(web_stats)

# print(df.head())

df.set_index('Day', inplace = True)

 # print(df.index)

df.Visitors.plot()

# plt.show()

print(df[['Visitors','Bounce_Rate']])

ex_list = df.Visitors.tolist()
print(ex_list)