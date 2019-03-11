"""Download data needed for the examples"""

from __future__ import print_function

if __name__ == "__main__":

    from os import path, makedirs, remove
    from download_sample_data import bar as progressbar
    
    import pandas as pd
    import numpy as np
    import sys
    
    try:
        import requests
    except ImportError:
        print('Download script required requests package: conda install requests')
        sys.exit(1)
    
    def _download_dataset(url):
        r = requests.get(url, stream=True)
        output_path = path.split(url)[1]
        with open(output_path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progressbar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
    
    examples_dir = path.dirname(path.realpath(__file__))
    data_dir = path.join(examples_dir, 'data')
    if not path.exists(data_dir):
        makedirs(data_dir)
    
    # Taxi data
    def latlng_to_meters(df, lat_name, lng_name):
        lat = df[lat_name]
        lng = df[lng_name]
        origin_shift = 2 * np.pi * 6378137 / 2.0
        mx = lng * origin_shift / 180.0
        my = np.log(np.tan((90 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
        my = my * origin_shift / 180.0
        df.loc[:, lng_name] = mx
        df.loc[:, lat_name] = my
    
    taxi_path = path.join(data_dir, 'nyc_taxi.csv')
    if not path.exists(taxi_path):
        print("Downloading Taxi Data...")
        url = ('https://storage.googleapis.com/tlc-trip-data/2015/'
               'yellow_tripdata_2015-01.csv')
    
        _download_dataset(url)
        df = pd.read_csv('yellow_tripdata_2015-01.csv')
    
        print('Filtering Taxi Data')
        df = df.loc[(df.pickup_longitude < -73.75) &
                    (df.pickup_longitude > -74.15) &
                    (df.dropoff_longitude < -73.75) &
                    (df.dropoff_longitude > -74.15) &
                    (df.pickup_latitude > 40.68) &
                    (df.pickup_latitude < 40.84) &
                    (df.dropoff_latitude > 40.68) &
                    (df.dropoff_latitude < 40.84)].copy()
    
        print('Reprojecting Taxi Data')
        latlng_to_meters(df, 'pickup_latitude', 'pickup_longitude')
        latlng_to_meters(df, 'dropoff_latitude', 'dropoff_longitude')
        df.rename(columns={'pickup_longitude': 'pickup_x', 'dropoff_longitude': 'dropoff_x',
                           'pickup_latitude': 'pickup_y', 'dropoff_latitude': 'dropoff_y'},
                  inplace=True)
        df.to_csv(taxi_path, index=False)
        remove('yellow_tripdata_2015-01.csv')
        
    
    print("\nAll data downloaded.")
    
