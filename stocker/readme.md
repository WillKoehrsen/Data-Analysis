# Stocker: A Stock Analysis and Prediction Toolkit using Additive Models 

Stocker can be run from an interative Python 3.6 session. I recommend 
installing the [Anaconda Python 3.6 distribution](https://www.anaconda.com/download/)
and using a Jupyter Notebook. 

Stocker is a work in progress. Let me know of any issues and feel 
free to make your own contributions! 

## Requirements 

Python 3.6 and the following packages are required:

	quandl 3.3.0
	matplotlib 2.1.1
	numpy 1.14.0
	fbprophet 0.2.1
	pystan 2.17.0.0
	pandas 0.22.0
	pytrends 4.3.0

These can be installed with pip from the command line
(some of these might require running the command prompt as 
administrator). 

`pip install -U quandl numpy pandas fbprophet matplotlib pytrends pystan`

If pip does not work and you have the Anaconda 
distribution, try installing with conda:

`conda install quandl numpy pandas matplotlib pystan`

`conda update quandl numpy pandas matplotlib pystan`

pytrends and fbprophet can only be installed with pip. If you run into 
any other errors installing packages, check out [Stack Overflow](https://stackoverflow.com/)

You may try `conda install -c conda-forge fbprophet` to install with conda.

## Getting Started

Once the required packages have been installed, get started exploring a stock 
by running an interactive python session or Jupyter Notebook in the same
folder as stocker.py. 

Import the stocker class by running

`from stocker import Stocker`

Instantiate a stocker object by calling Stocker with a valid stock ticker. Stocker uses
the WIKI database on Quandl by default and a list of all 3100
tickers in this database can be found at data/stock_list.csv. 
If using one of the tickers in the list, only the ticker symbol needs to be passed. 
If using a stock not on the list, try using a different exchange:

	# MSFT is in the WIKI database, which is default
	microsoft = Stocker(ticker='MSFT')
	
	# TECHM is in the NSE database
	techm = Stocker(ticker='TECHM', exchange='NSE')

If succesful, you will recieve a message with the date range of data:

`MSFT Stocker Initialized. Data covers 1986-03-13 to 2018-01-12.`

# Methods

The Stocker object includes 8 main methods for analyzing and predicting 
stock prices. Call any of the following on your stocker object, replacing
`Stocker` with your object (for example `microsoft`):

### Plot stock history

`Stocker.plot_stock(start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic')`
​	
Prints basic info for the specified stats and plots the history for the stats
over the specified date range. The default stat is Adjusted Closing price and 
default start and end dates are the beginning and ending dates
of the data. `plot_type` can be either basic, to plot the actual values on the 
y-axis, or `pct` to plot percentage change from the average. 

### Make basic prophet model

`model, future = Stocker.create_prophet_model(days=0, resample=False)`

The number of training years for any Prophet model can be set with the 
`Stocker.training_years` attribute. The default number of training years is 3.

Make a Prophet Additive Model using the specified number of training years
and make predictions number of days into the future. If days > 0, prints the 
predicted price. Also plots the historical data with the predictions and uncertainty overlaid.
Returns the prophet model, and the future dataframe which can be used 
for plotting components of the time series. 

To see the trends and patterns of the prophet model, call 

`import matplotlib.pyplot as plt
model.plot_components(future)
plt.show()`

### Find significant changepoints and try to correlate with Google search trends

`Stocker.changepoint_date_analysis(search=None)`

Finds the most significant changepoints in the dataset from a prophet model trained 
using the assigned years of training data. The changepoints represent where the change in the
rate of change of the data is the greatest in either the negative or positive
direction. The changepoints occur where the change in the rate of the time series is greatest.
This method prints the 5 most significant changepoints ranked by the 
change in the rate and plots the 10 most significant overlaid on top of the 
stock price data. The changepoints only come from the first 80% of the training data in 
a Prophet model.

A special bonus feature of this method is a Google Search Trends analysis. If a search term is 
passed to the method, the method retrieves the Google Search Frequency for the specified term and plots
on the same graph as the changepoints and the stock price data. It also displays related 
search queries and related rising search queires. If no 
term is specified then this capability is not used. You can use
this to determine if the stock price is correlated to certain search terms or if the 
changepoints coincide with an increase in particular searches. 

### Calculate profit from buy and hold strategy

`Stocker.buy_and_hold(start_date=None, end_date=None, nshares=1)`

Evaluates a buy and hold strategy from the start date to the end date
with the specified number of shares. If no start date and end date are 
specified, these default to the start and end date of the data. The buy and
hold strategy means buying the stock on the start date and holding to the end date
when we sell the stock. Prints the expected profit and plots the profit over time. 
Recommended for those planning a trip back in time to maximize profits. 

### Find the best changepoint prior scale graphically

`Stocker.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2], 
olors=['b', 'r', 'grey', 'gold'])`

Makes a prophet model with each of the specified changepoint prior scales (cps).
The cps controls the amount of overfitting in the model: a higher cps means a more
flexible model which can lead to overfitting the training data (more variance), 
and a lower cps leads to less flexibility and the possiblity of underfitting (high bias).
Each model is fit with the assigned number of years of data and makes predictions for 6 months.
Output is a graph showing the original observations, with the predictions from each model 
and the associated uncertainty.

The cps is an attribute of a stocker object and can be changed using `Stocker.changepoint_prior_scale` 
The default value for the cps is 0.05 which tends to be low for fitting stock data. 

Altering the changepoint prior scale can have a significant effect on predictions,
so try a few different values to see how they affect the model.

### Quantitaively compare different changepoint prior scales

`Stocker.changepoint_prior_validation(self, start_date=None, end_date=None,
​				changepoint_priors = [0.001, 0.05, 0.1, 0.2])`

Quantifies the differences in performance on a validation set of the specified 
cps values. A model is created with each changepoint prior, trained on the assigned
number of training years prior to the test period and evaluated on the range
passed to the method. The default validation period is from two years before the end of the 
data to one year before the end of the data. The average error on the training and testing
data for each prior is calculated and displayed as well as the average uncertainty
(range) of the data for both the training and testing sets. The average error is the 
mean of the absolute difference between the prediction and the correct value in dollars.
The uncertainty is the upper estimate minus the lower estimate in dollars.
A graph of these results is also produced. This method is useful for choosing a 
proper cps in combination with the graphical results. 

### Evalaute the Prophet model predictions against real prices and play stock marker

`Stocker.evaluate_prediction(start_date=None, end_date=None, nshares=1000)`

Evalutes a trading strategy informed by the prophet model 
between the specified start and end date. The start and end date for the evaluation 
should be different than the start and end date used for validation the prior 
otherwise you could end up overfitting the test set. The model is trained on the assigned 
number of years of data prior to the test period and makes predictions for the specified date range. The 
default evaluation range is the last year of the data. Numerical performance metrics are computed
using the predictions and known test set values. These are: average absolute error on the testing
and training data, percentage of time the model predicted the correct direction for the stock, and the
percentage of the time the actual value was within the 80% confidence interval for the prediction. A 
graph shows the predictions with uncertainty and the actual values. The final actual and predicted
prices are also displayed. 

If number of shares is passed to the method, we get to play the stock market over the 
testing period with the specified number of shares. We compare the strategy informed 
by the Prophet model with a simple buy and hold approach. 

The strategy from the model states that for a given  day, we buy a stock if the model 
predicts it will increase. If the model predicts a decrease, we do not play the market on that day. 
Our earnings, if we bought the stock, will be the change in the price of the stock over that day
multiplied by the number of shares.  Therefore, if we predict the stock will go up and the price 
does go up, we will make the change in price times the number of shares. If the price goes down, 
we lose the change in price times the number of shares. 

Printed output is the final predicted price, the final actual price, the 
profit from the model strategy, and the profit from a buy and hold strategy over the 
same period. A graph of the expected profit from both strategies over time is displayed. 

### Predict future prices

`Stocker.predict_future(days=30)`

Makes a prediction for the specified number of days in the future 
using a prophet model trained on the assigned number of years of data. Printed output 
is the days on which the stock is expected to increase and the days when it is expected to decrease.
A graph also shows these results with confidence intervals for the prediction. 
​	