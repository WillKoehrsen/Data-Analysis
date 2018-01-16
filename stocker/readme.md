# Stocker: A Stock Analysis and Predictive using Additive Models Toolkit

Stocker is designed to be run from an interative Python 3.6 session. 
I recommended using a Jupyter Notebook. 

Python 3.6 is required. The following packages are required:

quandl 3.3.0
matplotlib 2.1.1
numpy 1.14.0
fbprophet 0.2.1
pandas 0.22.0
pytrends 4.3.0

These can all be installed with pip from the command line
(some of these might require running the command line as 
administrator)

pip install -U quandl numpy pandas fbprophet matplotlib pytrends

Once the packages have been installed, get started exploring a stock 
by running an interactive python session or Jupyter Notebook in the same
folder as stocker.py. 

Import the stocker class by running

from stocker import Stocker

Instantiate a stocker object by calling Stocker with a valid stock ticker:

microsoft = Stocker('MSFT')

If succesful, you will recieve a message with date range of data:

MSFT Stocker Initialized. Data covers 1986-03-13 to 2018-01-12.

The Stocker object includes 8 main methods for analyzing and predicting 
stock prices.

	* plot_stock(start_date=None, end_date=None)
	Prints basic information and plots the history of the stock. The 
	default start and end dates are the extent of the data
	* changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2], 
		colors=[])