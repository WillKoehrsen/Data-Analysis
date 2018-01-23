
# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# fbprophet for additive models
import fbprophet

# gspread for Google Sheets access
import gspread

# slacker for interacting with Slack
from slacker import Slacker

# oauth2client for authorizing access to Google Sheets
from oauth2client.service_account import ServiceAccountCredentials

# os for deleting images
import os

# matplotlib for plotting 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

# import weighter
from weighter import Weighter

if __name__ == "__main__":

	# google sheets access
	scope = ['https://spreadsheets.google.com/feeds']

	# Use local stored credentials in json file
	# make sure to first share the sheet with the email in the json file
	credentials = ServiceAccountCredentials.from_json_keyfile_name('C:/Users/Will Koehrsen/Desktop/weighter-2038ffb4e5a6.json', scope)

	# Authorize access
	gc = gspread.authorize(credentials);

	# Slack api key is stored as text file
	with open('C:/Users/Will Koehrsen/Desktop/slack_api.txt', 'r') as f:
	    slack_api_key = f.read()

	slack = Slacker(slack_api_key)

	# Open the sheet, need to share the sheet with email specified in json file
	gsheet = gc.open('Auto Weight Challenge').sheet1

	# List of lists with each row in the sheet as a list
	weight_lists = gsheet.get_all_values()

	# Headers are the first list
	# Pop returns the element (list in this case) and removes it from the list
	headers = weight_lists.pop(0)

	# Convert list of lists to a dataframe with specified column header
	weights = pd.DataFrame(weight_lists, columns=headers)

	# Record column should be a boolean
	weights['Record'] = weights['Record'].astype(bool)

	# Name column is a string
	weights['Name'] = weights['Name'].astype(str)

	# Convert dates to datetime, then set as index, then set the time zone
	weights['Date'] = pd.to_datetime(weights['Date'], unit='s')
	weights  = weights.set_index('Date', drop = True).tz_localize(tz='US/Eastern')

	# Drop any extra entries
	weights = weights.drop('NaT')

	# If there are new entries create the weighter object
	if len(weights) > np.count_nonzero(weights['Record']):
		# Initialize with dataframe of weights, google sheet, and slack object
    	 weighter = Weighter(weights, gsheet, slack)
    	 weighter.process_entries()
    	 print('Success')


	

