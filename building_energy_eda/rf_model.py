# Script Name: rf_model.py
# Purpose: Implementation of machine learning functions written in model_utilities.py
# Authors: William Koehrsen
# License: Creative Commons Attribution-ShareAlike 4.0 International License.
##########
# Latest Changelog Entries:
# v0.00.01 - 11/15/17 - rf_model.py - William Koehrsen began this script
# v0.00.02 - 11/18/17 - rf_model.py - William Koehrsen completed the command line parser arguments
# v0.00.03 - 11/23/17 - rf_model.py - William Koehrsen finished using the model for training and testing random forest on data
# v0.00.04 - 11/25/17 - rf_model.py - William Koehrsen finished using the model for challenging methods used in prediction
##########

import argparse
import numpy as np
import feather 

from model_utilities import get_features_labels, train_random_forest, mape 
from model_utilities import complete_model, predict_mid, training_curves

# Create an argument parser
parser = argparse.ArgumentParser(description = 'Process a building name')
parser.add_argument('buildingName', metavar = 'BLDG', type = str, help = "A string representing the name of the buildling")

# Parse the arguments and extract the building name
args = parser.parse_args()
building_name = args.buildingName

# Create the filepath
filepath = ('python_data/f-%s_weather.csv' % building_name)

# Complete model for each season
results_df, predictions_df, features_df = complete_model(filepath)

# Write the season results to feather files
feather.write_dataframe(results_df, 
	('feather/%s_season_performance_metrics.feather' % building_name))
feather.write_dataframe(predictions_df, 
	('feather/%s_season_predictions.feather' % building_name))
feather.write_dataframe(features_df, 
	('feather/%s_feature_importances.feather' % building_name))

# Midpoint model for each season
mid_df, metrics_mid_df = predict_mid(filepath)

# Write the midpoint results to feather files
feather.write_dataframe(metrics_mid_df, 
	('feather/%s_mid_performance_metrics.feather' % building_name))
feather.write_dataframe(mid_df, 
	('feather/%s_mid_feather' % building_name))

# Training curves for model
training_curve_df = training_curves(filepath)

# Write the training curve information to feather files
feather.write_dataframe(training_curve_df, 
	('feather/%s_training_curves.feather' % building_name))