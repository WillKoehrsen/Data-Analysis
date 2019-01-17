import os
from bs4 import BeautifulSoup
from dateutil import parser
from datetime import datetime
import pandas as pd


def process_bargraph(bargraph):
    bardata = [bar.get('data-tooltip')
               for bar in bargraph.find_all(attrs={'class': 'bargraph-bar'})]
    print(len(bardata))
    return
    # Sort by xposition
    bardata = sorted(bardata, key=lambda x: float(x.get('x')))
    views = [float(s.split(' ')[0].replace(',', '')) for s in bardata]
    dates = [s.split(' ')[-1].split('\xa0')[0] + ' '
             + s.split(' ')[-1].split('\xa0')[1] for s in bardata]
    year = str((datetime.now() - pd.Timedelta(days=i * 30)).year)
    dates = [parser.parse(d + ' ' + year) for d in dates]
    return views, dates


files = os.listdir('html_pages')

v = []
d = []

for fid in files:
    i = int(fid.split('.')[0].split('p')[1])
    graph = BeautifulSoup(
        open(f'html_pages/{fid}', 'r')).find_all(attrs={'class': 'bargraph'})[0]
    r = process_bargraph(graph, i)
    v.extend(r[0])
    d.extend(r[1])
    results = pd.DataFrame({'date': d, 'views': v})

results['date'] = pd.to_datetime(results['date'])
results.to_parquet('medium_views_time')
