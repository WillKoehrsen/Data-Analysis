# Script Name: model_utilities.py
# Purpose: Machine learning data preparation and model implementation
# Authors: William Koehrsen
# License: Creative Commons Attribution-ShareAlike 4.0 International License.
##########
# Latest Changelog Entries:
# v0.00.01 - 11/02/17 - model_utilities.py - William Koehrsen began this file by transferring code from eda_report_two.ipynb
# v0.00.02 - 11/10/17 - model_utilities.py - William Koehrsen finished the code for evaluating models
# v0.00.03 - 11/10/17 - model_utilities.py - William Koehrsen finished the Random Forest model implementation
# v0.00.04 - 11/10/17 - model_utilities.py - William Koehrsen finished the Random Forest hyperparameter optimization
##########

# Rmd code goes below the comment marker!

import pandas as pd
import numpy as np
import os
import re
import random
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from dateutil import parser
from tpot import TPOTRegressor
import tensorflow as tf
import feather
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function that takes in a dataframe, season and returns the features, X, and labels, Y, as numpy arrays. 
# The function segments the data by the required season and can return the dates 
# associated with the data as well as the column names to use for determining feature importances. 

def get_features_labels(df, season = 'summer', analysis_col = 'forecast', return_dates = False, names = False, raw_df = False):

   # Get months and days for subsetting and use for additional features
    months = []
    days = []
    # Extract dates from dataframe
    dates = list(df['timestamp'])
    # Create list of only month
    [months.append(date.split('-')[1]) for date in dates]
    months = list(map(int, months))
    ## Create list of only days
    [days.append(date.split('-')[2][0:2]) for date in dates]
    days = list(map(int, days))
    # Add the months and days to the dataframe
    df['month'] = months
    df['day'] = days
    
    # Define seasons
    spring = [3, 4, 5]
    summer = [6, 7, 8]
    fall = [9, 10, 11]
    winter = [12, 1, 2]
    
    # Extract only the relevant months
    if season.lower() == 'spring':
        df = df[df['month'].isin(spring)]
    elif season.lower() == 'summer':
        df = df[df['month'].isin(summer)]
    elif season.lower() == 'fall':
        df = df[df['month'].isin(fall)]
    elif season.lower() == 'winter':
        df = df[df['month'].isin(winter)]
    else:
        print('Choose a valid season')
        return
    
    # Convert cyclical features
    df['num_time_sin'] = np.sin(df['num_time'] * (2 * np.pi/24))
    df['num_time_cos'] = np.cos(df['num_time'] * (2 * np.pi/24))

    # Create list of datetimes for converting to seconds
    dt = [datetime.datetime.strptime(df.ix[index, 'timestamp'], 
        "%Y-%m-%d %H:%M:%S") for index in list(df.index)]

    min_time = min(dt)

    # Create column of time since start in seconds
    seconds = [(time.mktime(date.timetuple()) - 
        time.mktime(min_time.timetuple())) for date in dt]
    
    df['time_seconds'] = seconds

    # Create column of days since start of season
    df['season_day'] = ((df['month'] -min(df['month'])) * 30) + df['day']
    # Need to treat month 12 slightly differently
    # .ix used for mixed indexing, .iloc used for integer indexing, .loc used for name indexing
    df.ix[list(df[df['month'] == 12].index), 'season_day'] = df['day']

    df.ix[list(df[df['month'] == 12].index), 'month'] = 0

    # Extract the dates for later use in analysis and plotting
    dates = np.array(df['timestamp'])

    # Get targets: cleaned electricity or forecast electricity
    y = np.array(df[analysis_col])
    
    # Drop the columns that are not related to time or weather
    df = df.drop(['forecast', 'timestamp', 'elec_cons', 'elec_cons_imp', 'pow_dem', 'cleaned_energy', 'anom_flag', 'num_time','anom_missed_flag'], axis = 1)

    # One_hot encoding of categorical variables
    df = pd.get_dummies(df, columns = ['sun_rise_set', 'biz_day', 'day_of_week', 'week_day_end'])
    
    if raw_df:
        return df

    # Get columns to find feature importances
    col_names = list(df.columns)
    
    # Convert to np array
    df = np.array(df)
    
    # Create a min max scaler to get all features between 0 and 1
    scaler = MinMaxScaler()
    
    # Transform features to between 0 and 1
    X = scaler.fit_transform(df)
    
    # Return the features and labels as numpy arrays
    if names and return_dates:
        return X, y, dates, col_names
    
    elif names:
        return col_names
    
    elif return_dates:
        return X, y, dates
    
    else:
        return X, y

# Calculates the mean average percentage error between the true values and the predictions
def mape(y_true, predictions):
    y_true = np.array(y_true)
    predictions = np.array(predictions)
    # Find where y true is not equal to zero to avoid division by error infinities/errors
    mean_diff = np.mean(np.where(y_true >0, np.divide(abs(y_true - predictions), y_true) * 100, 0))
    return mean_diff

# Function takes in a datafame and season and trains a random forest model on the data
def train_random_forest(df, season = 'summer', analysis_col = 'forecast', return_pred = False, n_estimators = 200):
    
    # Get array of features and array of targets
    X, y = get_features_labels(df = df, season = season, analysis_col = analysis_col)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    # Create the model and fit on the training data, use 100 decision trees
    model = RandomForestRegressor(n_estimators = n_estimators, criterion = 'mse', max_depth = 40)
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    predictions = model.predict(X_test)
    
    # Find the performance metrics
    rmse = round(np.sqrt(mean_squared_error(y_test, predictions)), 6)
    r_squared = round(model.score(X_test, y_test), 6)
    mape_result = round(mape(y_test, predictions), 6)
    
    if return_pred:
        return model, rmse, r_squared, mape_result, predictions, y_test
    else:
        return model, rmse, r_squared, mape_result

# Plots all season predictions for a given model
def complete_model(filepath, analysis_col = 'forecast', plot = False):
    building_name = re.split('-|_', filepath)[1]
    results_dict = {}
    df = pd.read_csv(filepath)
    overall_df = pd.DataFrame()
    seasons = ['spring', 'summer', 'fall', 'winter']
    season_results_df = pd.DataFrame(columns = ['season', 'rmse', 'rsquared', 'mape'])

    # Iterate through all seasons and get performance metrics and predictions
    for index, season in enumerate(seasons):
        # Create two new dataframes for predictions and metrics
        predictions_df = pd.DataFrame()

        model, rmse, r_squared, mape_result, predictions, y_true = train_random_forest(df, analysis_col = analysis_col, season = season, return_pred = True)

        if index == 0:
            col_names = get_features_labels(pd.read_csv(filepath), names = True)

            feature_importances = np.array(model.feature_importances_)
            
        else:   
            feature_importances += model.feature_importances_
        # Add performance metrics to season_df
        season_results_df.loc[index] = [season, rmse, r_squared, mape_result]
        

        predictions_df['%s_pred' % season] = predictions
        predictions_df['%s_true' % season] = y_true
        

        # Concatenate the season performance metrics and predictions with a
        # dataframe to hold all seasons
        overall_df = pd.concat([overall_df, predictions_df], axis = 1)

        # Decent plots using matplotlib
        if plot:
            colors = ["green", "red", "orange", "blue"]
            plt.figure(figsize = (10, 20))
            plt.subplot(4, 1, index + 1)
            plt.scatter(predictions, y_true, color = colors[index])
            plt.ylabel('True Value of Electricity Consumption');
            plt.xlabel('Prediction Value of Electricity Consumption');
            plt.title('%s %s True vs. Predicted Energy with Random Forest Model'% (building_name, season));
            print('R-Squared: {:0.4f} rmse: {:0.4f}'.format(r_squared, rmse))
            plt.show()

    average_feature_importances = feature_importances / len(seasons)

    feature_import_dict = {'feature': col_names, 'importance': list(average_feature_importances)}

    feature_import_df = pd.DataFrame(data = feature_import_dict)

    # Return the performance metrics, the predictions and true values, and feature importances
    return season_results_df, overall_df, feature_import_df

def predict_mid(filepath):
    df = pd.read_csv(filepath)
    results_df = pd.DataFrame()
    metrics_df = pd.DataFrame(columns = ['season', 'rmse' , 'rsquared', 'mape'])
    results_df['forecast'] = df['forecast']
    results_df['timestamp'] = df['timestamp']
    
    seasons = ["spring", "summer", "fall", "winter"]

    # Predicts middle 25% of data for each season
    for index, season in enumerate(seasons):
        
        X, y, season_dates, names = get_features_labels(df, season = season, return_dates = True, names = True)
        month_index = names.index('month')
        
        # Take middle 25% of data for testing
        start_index = round(0.375 * y.shape[0])
        stop_index = round(0.625 * y.shape[0])

        # Join arrays together adding rows
        X_train_beginning = X[0:start_index]
        X_train_end = X[stop_index:]
        X_train = np.concatenate((X_train_beginning, X_train_end), axis = 0)
        X_test = X[start_index: stop_index]
        
        # Do same with features
        y_train_beginning = y[0:start_index]
        y_train_end = y[stop_index:]
        y_train = np.concatenate((y_train_beginning, y_train_end), axis = 0)
        y_test = y[start_index:stop_index]
        
        model = RandomForestRegressor(n_estimators = 200, criterion = 'mse', 
            max_depth = 40)
        
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        predictions_df = pd.DataFrame()
        predictions_df['timestamp'] = season_dates[start_index:stop_index]
        predictions_df[season] = predictions
        
        results_df = pd.merge(results_df, predictions_df, how = 'outer', on = 'timestamp')
        results_df[season] = np.nan_to_num(results_df[season])
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape_result = mape(y_test, predictions)
        rsquared = r2_score(y_test, predictions)

        metrics_df.loc[index] = [season, rmse, rsquared, mape_result]

        # print('Season {}; RMSE on test data: {:0.4f}'.format(season, rmse))
        
        # x_values = range(0, len(y_train) + len(y_test))
        
        # if season.lower() == "spring":
        #     spring_df = pd.DataFrame()
        #     spring_df['timestamp'] = [parser.parse(date) for date in season_dates[start_index:stop_index]]
        #     spring_df['predictions'] = predictions
        #     datetimes = pd.date_range('2016-03-01 00:00:00', periods = len(x_values), freq = '0.25H').tolist()
        # elif season.lower() == "summer":
        #     summer_df = pd.DataFrame()
        #     summer_df['timestamp'] = [parser.parse(date) for date in season_dates[start_index:stop_index]]
        #     summer_df['predictions'] = predictions
        #     datetimes = pd.date_range('2016-06-01 00:00:00', periods = len(x_values), freq = '0.25H').tolist()
        # elif season.lower() == "fall":
        #     fall_df = pd.DataFrame()
        #     fall_df['timestamp'] = [parser.parse(date) for date in season_dates[start_index:stop_index]]
        #     fall_df['predictions'] = predictions
        #     datetimes = pd.date_range('2016-08-01 00:00:00', periods = len(x_values), freq = '0.25H').tolist()
        # elif season.lower() == "winter":
        #     winter_df = pd.DataFrame()
        #     winter_df['timestamp'] = [parser.parse(date) for date in season_dates[start_index:stop_index]]
        #     winter_df['predictions'] = predictions
        #     datetimes = pd.date_range('2016-12-01 00:00:00', periods = len(x_values) , freq = '0.25H').tolist()
        
        # datetimes_test = datetimes[start_index:stop_index]
        
        # fig, ax = plt.subplots(figsize = (15, 6))
        
        # datemin = datetimes[0]
        # datemax = datetimes[-1]
        
        # ax.plot(datetimes, y, color = "black")
        # ax.plot(datetimes_test, predictions, color = "red")
        
        # yearsFmt = mdates.DateFormatter('%Y-%m')
        # ax.xaxis.set_major_formatter(yearsFmt)
        # ax.set_xlim(datemin, datemax)
        # fig.autofmt_xdate()
        
        # plt.legend(['Actual', 'Predicted'])
        # plt.xlabel('')
        # plt.ylabel('kWh')
        # plt.title('{} Predicted vs. Actual Energy Consumption'.format(season))
        
        # plt.show()
    
    # fig, ax = plt.subplots(figsize = (15, 10))
    # dt = [parser.parse(date) for date in results_df['timestamp']]
    # datemin = dt[0]
    # datemax = dt[-1]
    
    # ax.plot(dt, results_df['forecast'], color = 'black')
    # ax.plot(summer_df['timestamp'], summer_df['predictions'], color = 'yellow')
    # ax.plot(fall_df['timestamp'], fall_df['predictions'], color = 'orange')
    # ax.plot(winter_df['timestamp'], winter_df['predictions'], color = 'blue')
    # ax.plot(spring_df['timestamp'], spring_df['predictions'], color = 'green')
    
    # yearsFmt = mdates.DateFormatter('%Y-%m')
    # ax.xaxis.set_major_formatter(yearsFmt)
    # ax.set_xlim(datemin, datemax)
    # fig.autofmt_xdate()
    
    # plt.legend(['Actual', 'Predicted Summer', 'Predicted Fall', 'Predicted Winter', 'Predicted Fall'])
    # plt.xlabel('')
    # plt.ylabel('kWh')
    # plt.title('Predicted vs. Actual Energy Consumption')
    # plt.show()
    
    return results_df, metrics_df

def training_curves(filepath):
    # Function takes in a file path and returns training and testing scores
    # across a range of number of training points.
    # The data returned will be used for plotting training curves

    # Read in the data as a dataframe
    df = pd.read_csv(filepath)
    
    # Results dataframe to hold metrics
    results_df = pd.DataFrame(columns = ['season', 'set', 'rmse', 'rsquared',
     'mape_result', 'fraction', 'train_points', 'test_points'])
    
    # Index for adding to dataframe
    index = 0
    seasons = ['spring', 'summer', 'fall', 'winter']

    # Iterate through seasons for 20 different lengths of training data
    for season in seasons:

        X, y = get_features_labels(df, season = season)

        for fraction in np.arange(0.05, 1.05, 0.05):

            sample = np.random.permutation(len(X))
            # Length based on i from 0.05 to 1.00 in 0.05 increments
            subset_length = int(fraction * len(X))

            # Each time the X and y are randomly mixed and then subsetted
            X_mixed, y_mixed = X[sample][:subset_length], y[sample][:subset_length]
            
            # Train test split on the subset with the appropriate test size
            X_train_subset, X_test, y_train_subset, y_test = train_test_split(X_mixed, y_mixed, test_size = 0.3)

            # Length of training and testing datasets
            train_length = len(X_train_subset)
            test_length = len(X_test)

            # Create and train the model
            model = RandomForestRegressor(n_estimators = 200, max_depth = 40)
            model.fit(X_train_subset, y_train_subset)

            # Predict on the test data and evaluate training performance
            train_predictions = model.predict(X_train_subset)
            train_rmse = np.sqrt(mean_squared_error(y_train_subset, train_predictions))
            train_rsquared = r2_score(y_train_subset,  train_predictions)
            train_mape = mape(y_train_subset, train_predictions)
            train_length = len(X_train_subset)
            
            # Add training results to dataframe
            results_df.loc[index] = [season, 'train', train_rmse, 
            train_rsquared, train_mape, fraction, train_length, test_length]

            # Increment dataframe index
            index += 1
            
            # Predict on the test data and evaluate test performance
            test_predictions = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_rsquared = r2_score(y_test, test_predictions)
            test_mape = mape(y_test, test_predictions)
            test_length = len(X_test)
            
            # Add test results to dataframe
            results_df.loc[index] = [season, 'test', test_rmse, test_rsquared,
            test_mape, fraction, train_length, test_length]

            # Increment dataframe index
            index += 1

    # Convert point columns to floats to work with serialization in feather
    train_floats = [float(point) for point in results_df['train_points']]
    test_floats  = [float(point) for point in results_df['test_points']]

    # Add the columns back into the dataframe, replacing the integers
    results_df['train_points'] = train_floats
    results_df['test_points'] = test_floats

    return results_df
