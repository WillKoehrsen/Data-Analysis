# Data science imports
import pandas as pd
import numpy as np

# Options for pandas
pd.options.display.max_columns = 20

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Interactive plotting
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()

from timeit import default_timer as timer

from collections import Counter, defaultdict
from itertools import chain

from bs4 import BeautifulSoup
import re

import requests
from multiprocessing import Pool

def get_links(soup):
    """
    Retrieve all links to entries on webpage
    
    :param soup: BeautifulSoup of HTML for page
    :return entry_links: list of links to entries
    
    """    
    titles = soup.find_all(attrs = {'class': 'bq y br af bs ag db dc dd c de df dg'})
    pattern = re.compile('[0-9]{1,} min read')
    read_times = soup.find_all(text = pattern)
    read_times = [int(x.split(' ')[0]) for x in read_times]
    total_read_time = sum(read_times)
    
    print(f'Found {len(titles)} entries.')
    print(f'Total Read Time of Entries: {total_read_time} minutes.')
    entry_links = [title.a.get_attribute_list('href')[0] for title in titles]
    
    return entry_links

def process_entry(link):
    """
    Retrieve data of single entry.
    
    :param link: string for link to entry
    
    :return entry_dict: dictionary of data about entry
    """
    
    entry_dict = {}
     
    # Retrieve the article and create a soup
    entry = requests.get(link).content
    entry_soup = BeautifulSoup(entry, features="lxml")
    
    # Publication time
    t = entry_soup.find_all('time')[0]
    t = pd.to_datetime(t.get('datetime'), utc=True).tz_convert('America/New_York')

    # Find the title header (determines if an article or a response)
    if entry_soup.h1 is not None:
        title = entry_soup.h1.text
    else:
        title = f'response-{t}'

    # Text as single long string
    entry_text = [p.text for p in entry_soup.find_all('p')]
    entry_text = ' '.join(entry_text)

    # Word count
    word_count = len(entry_text.split(' '))

    # Reading time in minutes
    read_time = entry_soup.find_all(attrs={'class': 'readingTime'})
    read_mins = int(read_time[0].get('title').split(' ')[0])

    # Number of claps
    clap_pattern = re.compile('^[0-9]{1,} claps|^[0-9]{1,}.[0-9]{1,}K claps|^[0-9]{1,}K claps')
    claps = entry_soup.find_all(text = clap_pattern)

    if len(claps) > 0:
        if 'K' in claps[0]:
            clap_number = int(1e3 * float(claps[0].split('K')[0]))
        else:
            clap_number = int(claps[0].split(' ')[0])
    else:
        clap_number = 0

    # Post tags
    tags = entry_soup.find_all(attrs={'class': 'tags tags--postTags tags--borderless'})
    tags = [li.text for li in tags[0].find_all('li')]
        
    # Store in dictionary with title as key
    entry_dict['title'] = title
    entry_dict['text'] = entry_text
    entry_dict['word_count'] = word_count
    entry_dict['read_time'] = read_mins
    entry_dict['claps'] = clap_number
    entry_dict['time_published'] = t
    entry_dict['tags'] = tags
        
    
    return entry_dict
    
def process_in_parallel(links, processes=20):
    """
    Process entries in parallel
    
    :param links: list of entry links
    :param processes: integer number of processes (threads) to use in parallel
    
    :return results: list of dictionaries of entry data
    """
    pool = Pool(processes=processes)
    results = []

    start = timer()
    for i, result in enumerate(pool.imap_unordered(process_entry, links)):
        if (i + 1) % 5 == 0:
            print(f'{100 * i / len(links):.2f}% complete.', end='\r')
        results.append(result)

    pool.close()
    pool.join()
    end = timer()
    
    print(f'Processed {len(results)} entries in {end-start:.0f} seconds.')
    
    # Add extra columns with more data
    df = pd.DataFrame.from_dict(results)
    df['response'] = ['response' if x == True else 'article' for x in df['title'].str.contains('response')]
    df['claps_per_word'] = df['claps'] / df['word_count']
    df['words_per_minute'] = df['word_count'] / df['read_time']
    
    # Add 10 most common tags with flag if data has it
    n = 10
    all_tags = list(chain(*df['tags'].tolist()))
    tag_counts = Counter(all_tags)
    tags = tag_counts.most_common(n)

    for tag, count in tags:
        flag = [1 if tag in tags else 0 for tags in df['tags']]
        df.loc[:, f'<tag>{tag}'] = flag
        
    return df

def make_update_menu(base_title, article_annotations=None, response_annotations=None):
    """
    Make an updatemenu for interative plot
    
    :param base_title: string for title of plot
    
    :return updatemenus: a updatemenus object for adding to a layout
    """
    updatemenus = list([
    dict(
        buttons=list([
            dict(
                label='both', method='update', 
                args=[dict(visible=[True, True]), dict(title = base_title,
                                                       annotations=[article_annotations,response_annotations])]),
            dict(
                label='articles',
                method='update',
                args=[dict(visible=[True, False]), dict(title = 'Article ' + base_title,
                                                        annotations = [article_annotations])]),
            dict(
                label='responses',
                method='update',
                args=[dict(visible=[False, True]), dict(title='Response ' + base_title,
                                                       annotations = [response_annotations])]),
        ]))
    ])
    return updatemenus



def make_iplot(data, x, y, base_title, time=False, eq_pos=(0.75, 0.25)):
    """
    Make an interactive plot. Adds a dropdown to separate articles from responses
    if there are responses in the data. If there is only articles (or only responses)
    adds a linear regression line. 
    
    :param data: dataframe of entry data
    :param x: string for xaxis of plot
    :param y: sring for yaxis of plot
    :param base_title: string for title of plot
    :param time: boolean for whether the xaxis is a plot
    :param eq_pos: position of equation for linear regression
    
    :return figure: an interactive plotly object for display
    
    """

    # Extract the relevant data
    responses = data[data['response'] == 'response'].copy()
    articles = data[data['response'] == 'article'].copy()

    if not responses.empty:
        # Create scatterplot data, articles must be first for menu selection
        plot_data = [
            go.Scatter(
                x=articles[x],
                y=articles[y],
                mode='markers',
                name='articles',
                text=articles['title'],
                marker=dict(color='blue', size=12)),
            go.Scatter(
                x=responses[x],
                y=responses[y],
                mode='markers',
                name='responses',
                marker=dict(color='green', size=12))
        ]
        
        if not time:
            annotations = {}
            for df, name in zip([articles, responses], 
                                ['articles', 'responses']):
                
                regression = stats.linregress(x=df[x], y=df[y])
                slope = regression.slope
                intercept = regression.intercept
                rvalue = regression.rvalue

                xi = np.array(range(int(df[x].min()), int(df[x].max())))
                
                line = xi*slope + intercept
                trace = go.Scatter(
                                  x=xi,
                                  y=line,
                                  mode='lines',
                                  marker=dict(color='blue' if name == 'articles' else 'green'), 
                                  line=dict(width=4, dash='longdash'),
                                  name=f'{name} linear fit'
                                  )

                annotations[name] = dict(
                                  x=max(xi) * eq_pos[0],
                                  y=df[y].max() * eq_pos[1],
                                  showarrow=False,
                                  text=f'$R^2 = {rvalue:.2f}; Y = {slope:.2f}X + {intercept:.2f}$',
                          font=dict(size=16, color='blue' if name == 'articles' else 'green')
                          )

                plot_data.append(trace)
        
        # Make a layout with update menus
        layout = go.Layout(annotations=list(annotations.values()),
            height=600,
            width=900,
            title=base_title,
            xaxis=dict(
                title=x.title(),
                tickfont=dict(size=14),
                titlefont=dict(size=16)),
            yaxis=dict(
                title=y.title(),
                tickfont=dict(size=14),
                titlefont=dict(size=16)),
            updatemenus=make_update_menu(base_title, annotations['articles'], annotations['responses']))

    # If there are only articles
    else:
        plot_data = [
            go.Scatter(
                x=data[x],
                y=data[y],
                mode='markers',
                name = 'observations',
                text=data['title'],
                marker=dict(color='blue', size=12))
        ]
        
        regression = stats.linregress(x=data[x], y=data[y])
        slope = regression.slope
        intercept = regression.intercept
        rvalue = regression.rvalue
        
        xi = np.array(range(int(data[x].min()), int(data[x].max())))
        line = xi*slope + intercept
        trace = go.Scatter(
                          x=xi,
                          y=line,
                          mode='lines',
                          marker=dict(color='red'), 
                          line=dict(width=4, dash='longdash'),
                          name='linear fit'
                          )
        
        annotations = [dict(
                          x=max(xi) * eq_pos[0],
                          y=data[y].max() * eq_pos[1],
                          showarrow=False,
                          text=f'$R^2 = {rvalue:.2f}; Y = {slope:.2f}X + {intercept:.2f}$',
                  font=dict(size=16)
                  )]
        
        plot_data.append(trace)

        layout = go.Layout(annotations=annotations,
            height=600,
            width=900,
            title=base_title,
            xaxis=dict(
                title=x.title(),
                tickfont=dict(size=14),
                titlefont=dict(size=16)),
            yaxis=dict(
                title=y.title(),
                tickfont=dict(size=14),
                titlefont=dict(size=16)))

    # Add a rangeselector and rangeslider for a data xaxis
    if time:
        rangeselector = dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ]))
        rangeslider = dict(visible=True)
        layout['xaxis']['rangeselector'] = rangeselector
        layout['xaxis']['rangeslider'] = rangeslider
        
        figure = go.Figure(data=plot_data, layout=layout)
           
        return figure
        
    
    # Return the figure
    figure = go.Figure(data=plot_data, layout=layout)

    return figure