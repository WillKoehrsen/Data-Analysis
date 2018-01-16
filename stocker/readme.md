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

These can all be installed with pip from the command line
(some of these might require running the command prompt as 
administrator). 

`pip install -U quandl numpy pandas fbprophet matplotlib pytrends pystan`

If pip does not work and you have the Anaconda 
distribution, try installing with conda:

`conda install quandl numpy pandas fbprophet matplotlib pytrends pystan`
`conda update quandl numpy pandas fbprophet matplotlib pytrends pystan`

## Getting Started

Once the packages have been installed, get started exploring a stock 
by running an interactive python session or Jupyter Notebook in the same
folder as stocker.py. 

Import the stocker class by running

`from stocker import Stocker`

Instantiate a stocker object by calling Stocker with a valid stock ticker:

`microsoft = Stocker('MSFT')`

If succesful, you will recieve a message with date range of data:

`MSFT Stocker Initialized. Data covers 1986-03-13 to 2018-01-12.`

# Methods

The Stocker object includes 8 main methods for analyzing and predicting 
stock prices. Call any of the following on your stocker object, replacing
`Stocker` with your object (for example `microsoft`):

`Stocker.plot_stock(start_date=None, end_date=None)`
	
Prints basic information and plots the history of the stock. The 
default start and end dates are the extent of the data

`Stocker.buy_and_hold(start_date=None, end_date=None, nshares=1)`

Evaluates a buy and hold strategy from the start date to the end date
with the specified number of shares. If no start date and end date are 
specified, these default to the start and end date of the data. The buy and
hold strategy, besides being the smartest choice, is also the simplest. 
We buy at the start date and hold to the end date. Prints the expected
profit and plots the expected profit over time. 

`model, future = Stocker.create_prophet_model(resample=False)`

Make a Prophet Additive Model using 3 years of training data
and make predictions for 1 year into the future. Prints the 
predicted price for 1 year out and plots the historical 
data with the predictions and uncertainty overlaid.

Returns model, the prophet model, and future, the future dataframe.
To see the trends and patterns of the prophet model, call 

`import matplotlib.pyplot as plt
model.plot_components(future)
plt.show()`

`Stocker.changepoint_date_analysis(term=None)`

Finds the most significant changepoints in the dataset from a prophet model 
using the past 3 years of data. The changepoints represent where the change in the
rate of change of the data is the greatest in either the negative or positive
direction. In other words, a changepoint is where the second derivative of the data 
is at a maximum (if that is confusing, just think of it where the data goes from
increasing to decreasing or from increasing slowly to increase at a really
fast rate). This method prints the 5 most significant changepoints by the 
change in the rate of change and plots them overlaid on top of the 
stock price data. The changepoints only come from the first 80% of the training data.

A special bonus feature of this method is a Google Search Trends analysis. The 
method retrieves the Google Search Frequency for the specified term and plots
these on the same graph as the changepoints and the stock price data. If no 
term is specified, the term default to "ticker stock". You can use
this to determine if the stock price is related to certain search terms or if the 
changepoints coincide with particular searches. 

`Stocker.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2], 
olors=['b', 'r', 'grey', 'gold'])`

Makes a prophet model with each of the specified changepoint prior scales (cps).
The cps controls the amount of overfitting in the model: a higher cps means a more
flexible model which can lead to overfitting the training data (more variance), 
and a lower cps leads to less flexibility and the possiblity of underfitting (high bias).
Each model is fit with 3 years of data and makes predictions for 6 months. Output is 
a graph showing the original observations, with the predictions from each model 
and the associated uncertainty.

This may take a little while to run. The results can be used to select the best 
changepoint prior scale for the model. The cps is an attribute of a stocker object 
and can be changed using `Stocker.changepoint_prior_scale = 0.05` 

Altering the changepoint prior scale can have a significant effect on predictions,
so try a few different values to see how they alter models.

`Stocker.changepoint_prior_validation(changepoint_priors = [0.001, 0.05, 0.1, 0.2])`

Similar to the changepoint prior analysis except quantifies the differences between 
cps values. A model is created with each changepoint prior, trained on 3 years of 
data (2014-2016) and tested on 2017. The average error on the training and testing
data for each prior is calculated and displayed as well as the average uncertainty
(range) of the data for both the training and testing sets. The average error is the 
mean of the absolute difference between the prediction and the correct value in dollars,
and the uncertainty is the upper estimate minus the lower estimate in dollars as well.
A graph of these results is also produced. This method is useful for choosing
a proper cps in combination with the analysis graphical results. 

`Stocker.evaluate_prediction(nshares=1000)`

Evalutes a trading strategy informed by the prophet model for 
all of 2017. The model is trained on 3 years of data (2014-2016) 
and then makes predictions for 2017. These predictions are then compared
to the known stock price values to determine the profits (or losses) 
from using the prophet strategy. The strategy states that for a given 
day, we buy a stock if the model predicts it will increase. If the model predicts
it will decrease, we do not play the market on that day. Our profit, if we bought the 
stockfor the day is the change in the price of the stock over that day. Therefore,
if we predict the stock will go up, we buy the stock, and the price does go up, 
we will make that change in price times the number of shares. If however the price
goes down, we lose that change times the number of shares. 

Printed output is the final predicted price, the final actual price, the 
profit from the model strategy, and the profit from a buy and hold strategy over the 
same period. Graphs of the predictions versus the actual values and the expected 
profit from both strategies over time are also displayed. 

`Stocker.predict_future(days=30)`

Makes a prediction for the specified number of days in the future. 
Uses a prophet model trained on the past 3 years of data. Printed output 
is the final predicted value of the stock, the days on which the stock is 
expected to increase, and the days when it is expected to decrease.
A graph also shows these results with uncertainty intervals.
	
