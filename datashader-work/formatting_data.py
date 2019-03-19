


import geoviews as gv
import geoviews.feature as gf
import xarray as xr
from cartopy import crs

import pandas as pd
import numpy as np

gv.extension('bokeh', 'matplotlib')

xr_ensemble = xr.open_dataset('Data-Analysis/datashader-work/geoviews-examples/data/ensemble.nc').load()

from sqlalchemy import create_engine
engine = create_engine('postgres://localhost:5432/global_fishing_watch')
engine.table_names()
df = pd.read_sql("""SELECT * FROM fishing_effort LIMIT 10000""",
                engine, parse_dates=['date'])

df['flag'] = df['flag'].astype('category')
df['geartype'] = df['geartype'].astype('category')
df['lat'] = df['lat_bin'] / 100
df['lon'] = df['lon_bin'] / 100
df.info()



def format_df(df, n=10_000):
    ...:     df = df.iloc[:n]
    ...:     df = df.drop_duplicates(subset=['lat', 'lon', 'date'])
    ...:     df = df.sort_values(['lat', 'lon', 'date'])
    ...:     index = pd.MultiIndex.from_arrays([df['lat'], df['lon'], df['date']])
    ...:     df.index = index
    ...:     latitudes = df.index.levels[0]
    ...:     longitudes = df.index.levels[1]
    ...:     times = df.index.levels[2]
    ...:     return latitudes, longitudes, times, df