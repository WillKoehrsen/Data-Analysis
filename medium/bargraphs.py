from bs4 import BeautifulSoup
from selenium import webdriver
from dateutil import parser
from datetime import datetime, timedelta
import pandas as pd
import math
import time

driver = webdriver.Chrome()
driver.get("https://medium.com/me/stats")
input('Waiting for you to log in. Press enter when ready: ')
earliest_article_date = parser.parse(
    input('Enter earliest article date as string: ')).date()
days = (datetime.now().date()
        - earliest_article_date).total_seconds() / (60 * 60 * 24)
months = math.ceil(days / 30)


def get_all_pages(driver, xpath, months):

    # Initially starting at today
    latest_date_in_graph = datetime.now().date()

    print('Starting on ', latest_date_in_graph)

    views = []
    dates = []

    # Iterate through the graphs
    for m in range(months + 1):
        graph_views = []
        graph_dates = []
        # Extract the bar graph
        bargraph = BeautifulSoup(driver.page_source).find_all(
            attrs={'class': 'bargraph'})[0]

        # Get all the bars in the bargraph
        bardata = bargraph.find_all(attrs={'class': 'bargraph-bar'})
        # Sort the bar data by x position (which will be date order) with most recent first
        bardata = sorted(bardata, key=lambda x: float(
            x.get('x')), reverse=True)
        bardata = [bar.get('data-tooltip') for bar in bardata]
        latest_day = int(bardata[0].split('\xa0')[-1])

        # Some months are not overlapping
        if latest_day != latest_date_in_graph.day:
            latest_date_in_graph -= timedelta(days=1)

        # Iterate through the bars which now are sorted in reverse date order (newest to oldest)
        for i, data in enumerate(bardata):
            graph_views.append(float(data.split(' ')[0].replace(',', '')))
            graph_dates.append(latest_date_in_graph - timedelta(days=i))

        views.extend(graph_views)
        dates.extend(graph_dates)
        # Find the earliest date in the graph
        earliest_date_in_graph = graph_dates[-1]

        # Update the latest date in the next graph
        latest_date_in_graph = earliest_date_in_graph

        # Go to the previous graph
        driver.find_element_by_xpath(xpath).click()
        time.sleep(2)
        print(f'{100 * m /(months)}% complete.', end='\r')

    results = pd.DataFrame({'date': pd.to_datetime(
        dates), 'views': views}).groupby('date').sum()
    results = results.loc[results[results['views'] != 0.0].index.min():, ]
    print('First views on ', str(results.index.min().date()))
    return results


xpath = input('Paste xpath as string: ')

results = get_all_pages(driver, xpath, months)
fname = f'{str(datetime.now().date())}_stats'
results.to_parquet(fname)
print('Stats saved to ', fname)
