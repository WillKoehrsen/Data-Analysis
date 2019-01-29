from multiprocessing import Pool
import requests
import re
from bs4 import BeautifulSoup
from itertools import chain
from collections import Counter
from timeit import default_timer as timer
import pandas as pd
from datetime import datetime


def get_table_rows(fname='stats.html'):
    """
    Extract the table rows from the statistics

    :param fname: string name of the file stored in `data` directory

    :return table_rows: list of BeautifulSoup objects to be passed to `process_in_parallel`
    """

    soup = BeautifulSoup(
        open(f'data/{fname}', 'r', encoding='utf8'), features='lxml')
    table_rows = soup.find_all(
        attrs={'class': "sortableTable-row js-statsTableRow"})
    print(f'Found {len(table_rows)} entries in table.')
    return table_rows


def convert_timestamp(ts: int, tz: str):
    """Convert a unix timestamp to a date timestamp"""
    return pd.to_datetime(ts, origin='unix', unit='ms').tz_localize('UTC').tz_convert(tz).tz_localize(None)


def process_entry(entry, parallel=True, tz='America/Chicago'):
    """
    Extract data from one entry in table

    :param entry: BeautifulSoup tag
    :param parallel: Boolean for whether function is being run in parallel
    :param tz: string representing timezone for started and published time

    :return entry_dict: dictionary with data about entry

    """
    # Convert to soup when running in parallel
    if parallel:
        entry = BeautifulSoup(entry, features='lxml').body.tr

    entry_dict = {}
    # Extract information
    for value, key in zip(entry.find_all(attrs={'class': 'sortableTable-value'}),
                          ['published_date', 'views', 'reads', 'ratio', 'fans']):
        entry_dict[key] = float(
            value.text) if key == 'ratio' else int(value.text)

    entry_dict['read_time'] = int(entry.find_all(attrs={'class': 'readingTime'})[
                                  0].get('title').split(' ')[0])

    # Unlisted vs published
    entry_dict['type'] = 'unlisted' if len(
        entry.find_all(text=' Unlisted')) > 0 else 'published'

    # Publication
    publication = entry.find_all(attrs={'class': 'sortableTable-text'})
    if 'In' in publication[0].text:
        entry_dict['publication'] = publication[0].text.split('In ')[
            1].split('View')[0]
    else:
        entry_dict['publication'] = 'None'

    # Convert datetimes
    entry_dict['published_date'] = convert_timestamp(
        entry_dict['published_date'], tz=tz)
    entry_dict['started_date'] = convert_timestamp(
        entry.get('data-timestamp'), tz=tz)

    # Get the link
    link = entry.find_all(text='View story',
                               attrs={'class': 'sortableTable-link'})[0].get('href')
    entry_dict['link'] = link
    # Retrieve the article and create a soup
    entry = requests.get(link).content
    entry_soup = BeautifulSoup(entry, features='lxml')

    # Get the title
    try:
        title = entry_soup.h1.text
    except:
        title = 'response'

    title_word_count = len(re.findall(r"[\w']+|[.,!?;]", title))

    # Main text entries
    entry_text = [p.text for p in entry_soup.find_all(
        ['h1', 'h2', 'h3', 'p', 'blockquote'])]

    # Make sure to catch everything
    entry_text.extend(s.text for s in entry_soup.find_all(
        attrs={'class': 'graf graf--li graf-after--li'}))
    entry_text.extend(s.text for s in entry_soup.find_all(
        attrs={'class': 'graf graf--li graf-after--p'}))
    entry_text.extend(s.text for s in entry_soup.find_all(
        attrs={'class': 'graf graf--li graf-after--blockquote'}))
    entry_text.extend(s.text for s in entry_soup.find_all(
        attrs={'class': 'graf graf--li graf-after--pullquote'}))

    entry_text = ' '.join(entry_text)

    # Word count
    word_count = len(re.findall(r"[\w']+|[.,!?;]", entry_text))

    # Number of claps
    clap_pattern = re.compile(
        '^[0-9]{1,} claps|^[0-9]{1,}.[0-9]{1,}K claps|^[0-9]{1,}K claps')
    claps = entry_soup.find_all(text=clap_pattern)

    if len(claps) > 0:
        if 'K' in claps[0]:
            clap_number = int(1e3 * float(claps[0].split('K')[0]))
        else:
            clap_number = int(claps[0].split(' ')[0])
    else:
        clap_number = 0

    # Post tags
    tags = entry_soup.find_all(
        attrs={'class': 'tags tags--postTags tags--borderless'})
    tags = [li.text for li in tags[0].find_all('li')]

    # Responses to entry
    responses = entry_soup.find_all(attrs={'class': 'button button--chromeless u-baseColor--buttonNormal u-marginRight12',
                                           'data-action': 'scroll-to-responses'})
    num_responses = int(responses[0].text) if len(responses) > 0 else 0

    # Store in dictionary
    entry_dict['title'] = title
    entry_dict['title_word_count'] = title_word_count
    entry_dict['text'] = entry_text
    entry_dict['word_count'] = word_count
    entry_dict['claps'] = clap_number
    entry_dict['tags'] = tags
    entry_dict['num_responses'] = num_responses

    # Time since publication
    entry_dict['days_since_publication'] = (
        datetime.now() - entry_dict['published_date']).total_seconds() / (3600 * 24)

    return entry_dict


def process_in_parallel(table_rows, processes=20):
    """
    Process all the stats in a table in parallel

    :note: make sure to set the correct time zone in `process_entry`
    :note: running on Mac may first require setting
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    from the command line to enable parallel processing

    :param table_rows: BeautifulSoup table rows

    :param processes: integer number of processes (threads) to use in parallel

    :return df: dataframe of information about each post

    """
    # Convert to strings for multiprocessing
    table_rows_str = [str(r) for r in table_rows]

    # Process each article in paralllel
    pool = Pool(processes=processes)
    results = []
    start = timer()
    for i, r in enumerate(pool.imap_unordered(process_entry, table_rows_str)):
        # Report progress
        print(f'{100 * i / len(table_rows_str):.2f}% complete.', end='\r')
        results.append(r)
    pool.close()
    pool.join()
    end = timer()
    print(
        f'Processed {len(table_rows_str)} articles in {end-start:.2f} seconds.')

    # Convert to dataframe
    df = pd.DataFrame(results)
    # Rename ratio
    df.rename(columns={'ratio': 'read_ratio'}, inplace=True)
    # Add extra columns with more data
    df['claps_per_word'] = df['claps'] / df['word_count']
    df['editing_days'] = ((df['published_date'] - df['started_date']
                           ).dt.total_seconds() / (60 * 60 * 24)).astype(int)

    # Rounding
    df['published_date'] = df['published_date'].dt.round('min')
    df['started_date'] = df['started_date'].dt.round('min')
    df['read_ratio'] = df['read_ratio'].round(2)

    # 5 most common tags (might want to include more tags)
    n = 5
    all_tags = list(chain(*df['tags'].tolist()))
    tag_counts = Counter(all_tags)
    tags = tag_counts.most_common(n)

    # Adding columns with indication of tag
    for tag, count in tags:
        flag = [1 if tag in tags else 0 for tags in df['tags']]
        df.loc[:, f'<tag>{tag}'] = flag

    df.sort_values('published_date', inplace=True)
    return df


def get_data(fname='stats.html', processes=20):
    """
    Retrieve medium article statistics

    :note: running on Mac may first require setting
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    from the command line to enable parallel processing

    :param fname: file name (should be 'stats.html')
    :param processes: integer number of processes

    :return df: dataframe of article data
    """
    t = get_table_rows(fname=fname)
    return process_in_parallel(t, processes=processes)
