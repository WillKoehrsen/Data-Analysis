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


class Weighter():
    
    """
    When weighter is initialized, we need to convert the usernames,
    get a dictionary of the unrecorded entries, construct a dictionary
    of the actions to take, and make sure all data is formatted correctly
    """
    
    def __init__(self, weights, gsheet, slack):
        
        # Weights is a dataframe
        self.weights = weights.copy()

        self.gsheet = gsheet
        self.slack = slack

        
        # Users is a list of the unique users in the data
        self.users = list(set(self.weights['Name']))
        
        correct_names = []
        
        # Name Changes
        for user in self.weights['Name']:
            
            # Have to hardcode in name Changes
            if user == 'koehrcl':
                correct_names.append('Craig')
            elif user == 'willkoehrsen':
                correct_names.append('Will')
            elif user == 'fletcher':
                correct_names.append('Fletcher')
            
            # Currently do not handle new users
            else:
                print('New User Detected')
                return
            
        self.weights['Name'] = correct_names
        
        # Users is a list of the unique users in the data
        self.users = list(set(self.weights['Name']))
        
        # Create a dataframe of the unrecorded entries
        self.unrecorded = self.weights[self.weights['Record'] != True]
        
        # Process the unrecorded entries
        self.process_unrecorded()
        
        # The remaning entries will all be weights
        self.weights['Entry'] = [float(weight) for weight in self.weights['Entry']]
        
        # Build the user dictionary
        self.build_user_dict()
        
        # Calculate the change and percentage change columns
        self.calculate_columns()
        
    """ 
    Constructs a dictionary for each user with critical information
    This forms the basis for the summarize function
    """
    
    def build_user_dict(self):
        
        user_dict = {}
        
        user_goals = {'Craig': 215.0, 'Fletcher': 200.0, 'Will': 155.0}
        user_colors = {'Craig': 'forestgreen', 'Fletcher': 'navy', 'Will': 'darkred'}
        
        for i, user in enumerate(self.users):
            
            user_weights = self.weights[self.weights['Name'] == user]
            goal = user_goals.get(user)

            start_weight = user_weights.ix[min(user_weights.index), 'Entry']   
            start_date = min(user_weights.index)
            
            # Find minimum weight and date on which it occurs
            min_weight =  min(user_weights['Entry'])
            min_weight_date = ((user_weights[user_weights['Entry'] == min_weight].index)[0])
            
            # Find maximum weight and date on which it occurs
            max_weight = max(user_weights['Entry'])
            max_weight_date = ((user_weights[user_weights['Entry'] == max_weight].index)[0])
            
            most_recent_weight = user_weights.ix[max(user_weights.index), 'Entry']
            
            if goal < start_weight:
                change = start_weight - most_recent_weight
                obj = 'lose'
            elif goal > start_weight:
                change = most_recent_weight - start_weight
                obj = 'gain'
                
            pct_change = 100 * change / start_weight
            
            pct_to_goal = 100 * (change / abs(start_weight - goal) )
            
            # Color for plotting
            user_color = user_colors[user]
            
            user_dict[user] = {'min_weight': min_weight, 'max_weight': max_weight,
                               'min_date': min_weight_date, 'max_date': max_weight_date,
                               'recent': most_recent_weight, 'abs_change': change,
                               'pct_change': pct_change, 'pct_towards_goal': pct_to_goal,
                               'start_weight': start_weight, 'start_date': start_date,
                               'goal_weight': goal, 'objective': obj, 'color': user_color}
       
        self.user_dict = user_dict
             
    """
    Builds a dictionary of unrecorded entries where each key is the user
    and the value is a list of weights and methods called for by the user.
    This dictionary is saved as the entries attribute of the class.
    Removes the none weights from the data and from the google sheet.
    """
    
    def process_unrecorded(self):
        
        entries = {name:[] for name in self.users}
        drop = []
        
        location = {}
        
        for index in self.unrecorded.index:

            entry = self.unrecorded.ix[index, 'Entry']
            user = str(self.unrecorded.ix[index, 'Name'])
            
            # Try and except does not seem like the best way to handle this
            try:
                entry = float(entry)
                entries[user].append(entry)
                location[index] = True
                
            except:  
                entry = str(entry)
                entries[user].append(entry.strip())
                location[index] = 'remove'
                
                drop.append(index)
                
            self.weights.ix[index, 'Record'] = True
           
        # Indexes of new entries
        self.location = location
        
        # Update the Google Sheet before dropping
        self.update_sheet()
        
        # Drop the rows which do not contain a weight
        self.weights.drop(drop, axis=0, inplace=True)

        # Entries is all of the new entries
        self.entries = entries
        
    """ 
    Update the Google Spreadsheet. This involves removing the rows without weight
    entries and putting a True in the record column for all weights. 
    """

    def update_sheet(self):
        delete_count = 0
        
        # Iterate through the locations and update as appropriate
        for index, action in self.location.items():
            cell_row = (np.where(self.weights.index == index))[0][0] + 2 - delete_count
            if action == 'remove':
                self.gsheet.delete_row(index = cell_row)
                delete_count += 1
            elif action:
                self.gsheet.update_acell(label='D%d' % cell_row, val = 'True')
           
    """ 
    Iterates through the unrecorded entries and delegates 
    each one to the appropriate method.
    Updates the record cell in the google sheet 
    """
    def process_entries(self):
        for user, user_entries in self.entries.items():
            for entry in user_entries:
                
                # If a weight, display the basic message
                if type(entry) == float:
                    self.basic_message(user)
                    
                # If the message is a string hand off to the appropriate function
                else:
                    
                    # Require at lesat 8 days of data
                    if len(self.weights[self.weights['Name'] == user]) < 8:
                        message = "\nAt least 8 days of data required for detailed analysis."
                        self.slack.chat.post_message(channel='#weight_tracker', text = message, username = "Data Analyst", icon_emoji=":calendar:")
                
                    elif entry.lower() == 'summary':
                        self.summary(user)

                    elif entry.lower() == 'percent':
                        self.percentage_plot()

                    elif entry.lower() == 'history':
                        self.history_plot(user)

                    elif entry.lower() == 'future':
                        self.future_plot(user)

                    elif entry.lower() == 'analysis':
                        self.analyze(user)
    
                    # Display a help message if the string is not valid
                    else:
                        message = ("\nPlease enter a valid message:\n\n"
                                   "Your weight\n"
                                   "'Summary' to see a personal summary\n"
                                   "'Percent' to see a plot of all users percentage changes\n"
                                   "'History' to see a plot of your personal history\n"
                                   "'Future' to see your predictions for the next thirty days\n"
                                   "'Analysis' to view personalized advice\n"
                                   "For more help, contact @koehrsen_will on Twitter.\n")

                        self.slack.chat.post_message(channel='#weight_tracker', text = message, username = "Help", 
                        	icon_emoji=":interrobang:")
                    
            
    """ 
    Adds the change and percentage change columns to the self.weights df
    """
    def calculate_columns(self):
        
        self.weights = self.weights.sort_values('Name')
        self.weights['change'] = 0
        self.weights['pct_change'] = 0
        self.weights.reset_index(level=0, inplace = True)
        
        for index in self.weights.index:
            user = self.weights.ix[index, 'Name']
            weight = self.weights.ix[index, 'Entry']
            start_weight = self.user_dict[user]['start_weight']
            objective = self.user_dict[user]['objective']
            
            if objective == 'lose':
                
                self.weights.ix[index, 'change'] = start_weight - weight
                self.weights.ix[index, 'pct_change'] = 100 * (start_weight - weight) / start_weight
                
            elif objective == 'gain':
                self.weights.ix[index, 'change'] = weight - start_weight
                self.weights.ix[index, 'pct_change'] = 100 * (weight - start_weight) / start_weight

        self.weights.set_index('Date', drop=True, inplace=True)
        
                
    """ 
    This method is automatically run for each new weight
    """
    def basic_message(self, user):
    
        # Find information for user, construct message, post message to Slack
        user_info = self.user_dict.get(user)

        message = ("\n{}: Total Weight Change = {:.2f} lbs.\n\n"
                    "Percentage Weight Change = {:.2f}%\n").format(user, user_info['abs_change'],
                                                     user_info['pct_change'])

        self.slack.chat.post_message('#weight_tracker', text=message, username='Update', icon_emoji=':scales:')
                        
    """ 
    Displays comprehensive stats about the user
    """
    
    def summary(self, user):
        user_info = self.user_dict.get(user)
        message = ("\n{}, your most recent weight was {:.2f} lbs.\n\n"
                   "Absolute weight change = {:.2f} lbs, percentage weight change = {:.2f}%.\n\n"
                   "Minimum weight = {:.2f} lbs on {} and maximum weight = {:.2f} lbs on {}.\n\n"
                   "Your goal weight = {:.2f} lbs. and you are {:.2f}% of the way there.\n\n"
                   "You started at {:.2f} lbs on {}. Congratulations on the progress!\n").format(user, 
                     user_info['recent'], user_info['abs_change'], user_info['pct_change'], 
                     user_info['min_weight'], str(user_info['min_date'].date()),
                     user_info['max_weight'], str(user_info['max_date'].date()),
                     user_info['goal_weight'], user_info['pct_towards_goal'],                                                       
                     user_info['start_weight'], str(user_info['start_date'].date()))
        
        self.slack.chat.post_message('#weight_tracker', text=message, username='Summary', icon_emoji=":earth_africa:")
   
    """
    Reset the plot and institute basic parameters
    """
    @staticmethod
    def reset_plot():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams['text.color'] = 'k'
        
    """
    Plot of all users percentage changes.
    Includes polynomial fits (degree may need to be adjusted).
    """
    
    def percentage_plot(self):
        
        self.reset_plot()
        
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(10,8))

        for i, user in enumerate(self.users):
            
            user_color = self.user_dict[user]['color']

            # Select the user and order dataframe by date
            df = self.weights[self.weights['Name'] == user]
            df.sort_index(inplace=True)
            
            # List is used for fitting polynomial
            xvalues = list(range(len(df)))

            # Create a polynomial fit
            z = np.polyfit(xvalues, df['pct_change'], deg=6)

            # Create a function from the fit
            p = np.poly1d(z)

            # Map the x values to y values
            fit_data = p(xvalues)

            # Plot the actual points and the fit
            plt.plot(df.index, df['pct_change'], 'o', color = user_color, label = '%s Observations' % user)
            plt.plot(df.index, fit_data, '-', color = user_color, linewidth = 5, label = '%s Smooth Fit' % user)


        # Plot formatting
        plt.xlabel('Date'); plt.ylabel('% Change from Start')
        plt.title('Percentage Changes')
        plt.grid(color='k', alpha=0.4)
        plt.legend(prop={'size':14})
        plt.savefig('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\percentage_plot.png')
        
        self.slack.files.upload('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\percentage_plot.png', channels='#weight_tracker', title="Percent Plot")
        
        os.remove('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\percentage_plot.png')
        
    """ 
    Plot of a single user's history.
    Also plot a polynomial fit on the observations.
    """
    def history_plot(self, user):
        
        self.reset_plot()
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(10, 8))
        
        df = self.weights[self.weights['Name'] == user]
        df.sort_index(inplace=True) 
        user_color = self.user_dict[user]['color']
        
        # List is used for fitting polynomial
        xvalues = list(range(len(df)))

        # Create a polynomial fit
        z = np.polyfit(xvalues, df['Entry'], deg=6)

        # Create a function from the fit
        p = np.poly1d(z)

        # Map the x values to y values
        fit_data = p(xvalues)

        # Make a simple plot and upload to slack
        plt.plot(df.index, df['Entry'], 'ko', ms = 8, label = 'Observed')
        plt.plot(df.index, fit_data, '-', color = user_color, linewidth = 5, label = 'Smooth Fit')
        plt.xlabel('Date'); plt.ylabel('Weight (lbs)'); plt.title('%s Weight History' % user)
        plt.legend(prop={'size': 14});
        
        plt.savefig(fname='C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\history_plot.png')
        self.slack.files.upload('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\history_plot.png', channels='#weight_tracker', title="%s History" % user)
        
        # Remove the plot from local storage
        os.remove('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\history_plot.png')
   
    """ 
    Create a prophet model for forecasting and trend analysis.
    Might need to adjust model hyperparameters.
    """
    
    def prophet_model(self):
        model = fbprophet.Prophet(daily_seasonality=False, yearly_seasonality=False)
        return model
        
    """ 
    Plot the prophet forecast for the next thirty days
    Print the expected weight at the end of the forecast
    """
    def future_plot(self, user):
        self.reset_plot()
        
        df = self.weights[self.weights['Name'] == user]
        dates = [date.date() for date in df.index]
        df['ds'] = dates
        df['y'] = df['Entry']
        
        df.sort_index(inplace=True)

        # Prophet model
        model = self.prophet_model()
        model.fit(df)
        
        # Future dataframe for predictions
        future = model.make_future_dataframe(periods=30, freq='D')
        future = model.predict(future)
    
        color = self.user_dict[user]['color']
        
        # Write a message and post to slack
        message = ('{} Your predicted weight on {} = {:.2f} lbs.'.format(
            user, max(future['ds']).date(), future.ix[len(future) - 1, 'yhat']))
        
        self.slack.chat.post_message(channel="#weight_tracker", text=message, username = 'The Future', icon_emoji=":city_sunrise:")
        
        # Create the plot and upload to slack
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(df['ds'], df['y'], 'o', color = 'k', ms = 8, label = 'observations')
        ax.plot(future['ds'], future['yhat'], '-', color = color, label = 'modeled')
        ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], facecolor = color, 
                alpha = 0.4, edgecolor = 'k', linewidth  = 1.8, label = 'confidence interval')
        plt.xlabel('Date'); plt.ylabel('Weight (lbs)'); plt.title('%s 30 Day Prediction' % user)
        plt.legend()
        plt.savefig('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\future_plot.png')
        
        self.slack.files.upload('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\future_plot.png', channels="#weight_tracker", title="%s Future Predictions" % user)
        
        os.remove('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\future_plot.png')
        
    """ 
    Analyze user trends and provide recommendations. 
    Determine if the user is on track to meet their goal.
    """
    
    def analyze(self, user):
        
        self.reset_plot()
        
        # Get user info and sort dataframe by date
        info = self.user_dict.get(user)
        goal_weight = info['goal_weight']
        df = self.weights[self.weights['Name'] == user]
        df = df.sort_index()
        df['ds'] = [date.date() for date in df.index]
        df['y'] = df['Entry']
        
        model = self.prophet_model()
        model.fit(df)
        
        prediction_days = 2 * len(df)
        
        future = model.make_future_dataframe(periods = prediction_days, freq = 'D')
        future = model.predict(future)
        
        # lbs change per day 
        change_per_day = info['abs_change'] / (max(df['ds']) - min(df['ds'])).days
        
        days_to_goal = abs(int((info['recent'] - goal_weight) / change_per_day))
        date_for_goal = max(df['ds']) + pd.DateOffset(days=days_to_goal)
        
        # future dataframe where the user in above goal
        goal_future = future[future['yhat'] < goal_weight]
        
        # The additive model predicts the user will meet their goal
        if len(goal_future) > 0:
            model_goal_date = min(goal_future['ds'])
            message = ("\n{} Your average weight change per day is {:.2f} lbs\n"
                       "Extrapolating the average loss per day, you will reach your goal of {} lbs in {} days on {}.\n\n"
                       "The additive model predicts you will reach your goal on {}\n".format(
                       user, change_per_day, goal_weight, days_to_goal, date_for_goal.date(), model_goal_date.date()))
        
        # The additive model does not predict the user will meet their goal
        else:
            final_future_date = max(future['ds'])
            message = ("\n{} Your average weight change per day is {:.2f} lbs\n\n"
                       "Extrapolating the average loss per day, you will reach your goal of {} lbs in {} days on {}.\n\n"
                       "The additive model does not forecast you reaching your goal by {}.\n".format(
                           user, change_per_day, goal_weight, days_to_goal, date_for_goal.date(), final_future_date))
        
        
        
        self.slack.chat.post_message(channel="#weight_tracker", text=message, username="Analysis", icon_emoji=":bar_chart:")

        # Identify Weekly Trends
        future['weekday'] = [date.weekday() for date in future['ds']]
        future_weekly = future.groupby('weekday').mean()
        future_weekly.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
        
        # Color labels based on the users objective
        colors = ['red' if ( ((weight > 0) & (info['objective'] == 'lose')) | ((weight < 0) & (info['objective'] == 'gain'))) else 'green' for weight in future_weekly['weekly']]

        self.reset_plot()
        
        # Create a bar plot with labels for positive and negative changes
        plt.figure(figsize=(10, 8))
        xvalues = list(range(len(future_weekly)))
        plt.bar(xvalues, future_weekly['weekly'], color = colors, edgecolor = 'k', linewidth = 2)
        plt.xticks(xvalues, list(future_weekly.index))
        red_patch = mpatches.Patch(color='red',  linewidth = 2, label='Needs Work')
        green_patch = mpatches.Patch(color='green', linewidth = 2, label='Solid')
        plt.legend(handles=[red_patch, green_patch])
        plt.xlabel('Day of Week')
        plt.ylabel('Trend (lbs)')
        plt.title('%s Weekly Trends' % user)
        plt.savefig('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\weekly_plot.png')
        
        # Upload the image to slack and delete local file
        self.slack.files.upload('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\weekly_plot.png', channels = '#weight_tracker', title="%s Weekly Trends" % user)

        os.remove('C:\\Users\\Will Koehrsen\\Documents\\Data-Analysis\\weighter\\images\\weekly_plot.png')

        