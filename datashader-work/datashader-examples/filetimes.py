#!/usr/bin/env python3

"""
Simple test of read and write times for columnar data formats:
  python filetimes.py <filepath> [pandas|dask [hdf5base [xcolumn [ycolumn] [categories...]]]]

Test files may be generated starting from any file format supported by Pandas:
  python -c "import filetimes ; filetimes.base='<hdf5base>' ; filetimes.categories=['<cat1>','<cat2>']; filetimes.timed_write('<file>')"
"""

from __future__ import print_function

import time
global_start = time.time()

import os, os.path, sys, glob, argparse, resource, multiprocessing
import pandas as pd
import dask.dataframe as dd
import numpy as np
import datashader as ds
import bcolz
import feather
import fastparquet as fp

from datashader.utils import export_image
from datashader import transfer_functions as tf
from collections import OrderedDict as odict

#from multiprocessing.pool import ThreadPool
#dask.set_options(pool=ThreadPool(3)) # select a pecific number of threads
from dask import distributed

# Toggled by command-line arguments
DEBUG = False
DD_FORCE_LOAD = False
DASK_CLIENT = None

class Parameters(object):
    base,x,y='data','x','y'
    dftype='pandas'
    categories=[]
    chunksize=76668751
    cat_width=1 # Size of fixed-width string for representing categories
    columns=None
    cachesize=9e9
    parq_opts=dict(file_scheme='hive', has_nulls=False, write_index=False)
    n_workers=multiprocessing.cpu_count()


p=Parameters()

filetypes_storing_categories = {'parq'}


class Kwargs(odict):
    """Used to distinguish between dictionary argument values, and
    keyword-arguments.
    """
    pass

def benchmark(fn, args, filetype=None):
    """Benchmark when "fn" function gets called on "args" tuple.
    "args" may have a Kwargs instance at the end.
    If "filetype" is provided, it may be used to convert columns to
    categorical dtypes after reading (the "loading" is assumed).
    """
    posargs = list(args)
    kwargs = {}
    # Remove Kwargs instance at end of posargs list, if one exists
    if posargs and isinstance(posargs[-1], Kwargs):
        lastarg = posargs.pop()
        kwargs.update(lastarg)

    if DEBUG:
        printable_posargs = ', '.join([str(posarg.head()) if hasattr(posarg, 'head') else str(posarg) for posarg in posargs])
        printable_kwargs = ', '.join(['{}={}'.format(k, v) for k,v in kwargs.items()])
        print('DEBUG: {}({}{})'.format(fn.__name__, printable_posargs, ', '+printable_kwargs if printable_kwargs else '', flush=True))

    # Benchmark fn when run on posargs and kwargs
    start = time.time()
    res = fn(*posargs, **kwargs)

    # If we're loading data
    if filetype is not None:
        if filetype not in filetypes_storing_categories:
            opts=odict()
            if p.dftype == 'pandas':
                opts['copy']=False
            for c in p.categories:
                res[c]=res[c].astype('category',**opts)

        # Force loading (--cache=persist was provided)
        if p.dftype == 'dask' and DD_FORCE_LOAD:
            if DASK_CLIENT is not None:
                # 2017-04-28: This combination leads to a large drop in
                #   aggregation performance (both --distributed and
                #   --cache=persist were provided)
                res = DASK_CLIENT.persist(res)
                distributed.wait(res)
            else:
                if DEBUG:
                    print("DEBUG: Force-loading Dask dataframe", flush=True)
                res = res.persist()

    end = time.time()

    return end-start, res
    


read = odict([(f,odict()) for f in ["parq","snappy.parq","gz.parq","bcolz","feather","h5","csv"]])

def read_csv_dask(filepath, usecols=None):
    # Pandas writes CSV files out as a single file
    if os.path.isfile(filepath):
        return dd.read_csv(filepath, usecols=usecols)
    # Dask may have written out CSV files in partitions
    filepath_expr = filepath.replace('.csv', '*.csv')
    return dd.read_csv(filepath_expr, usecols=usecols)
read["csv"]          ["dask"]   = lambda filepath,p,filetype:  benchmark(read_csv_dask, (filepath, Kwargs(usecols=p.columns)), filetype)
read["h5"]           ["dask"]   = lambda filepath,p,filetype:  benchmark(dd.read_hdf, (filepath, p.base, Kwargs(chunksize=p.chunksize, columns=p.columns)), filetype)
def read_feather_dask(filepath):
    df = feather.read_dataframe(filepath, columns=p.columns)
    return dd.from_pandas(df, npartitions=p.n_workers)
read["feather"]      ["dask"] = lambda filepath,p,filetype:  benchmark(read_feather_dask, (filepath,), filetype)
read["bcolz"]        ["dask"]   = lambda filepath,p,filetype:  benchmark(dd.from_bcolz, (filepath, Kwargs(chunksize=1000000)), filetype)
read["parq"]         ["dask"]   = lambda filepath,p,filetype:  benchmark(dd.read_parquet, (filepath, Kwargs(index=False, columns=p.columns)), filetype)
read["gz.parq"]      ["dask"]   = lambda filepath,p,filetype:  benchmark(dd.read_parquet, (filepath, Kwargs(index=False, columns=p.columns)), filetype)
read["snappy.parq"]  ["dask"]   = lambda filepath,p,filetype:  benchmark(dd.read_parquet, (filepath, Kwargs(index=False, columns=p.columns)), filetype)
def read_csv_pandas(filepath, usecols=None):
    # Pandas writes CSV files out as a single file
    if os.path.isfile(filepath):
        return pd.read_csv(filepath, usecols=usecols)
    # Dask may have written out CSV files in partitions
    filepath_expr = filepath.replace('.csv', '*.csv')
    filepaths = glob.glob(filepath_expr)
    return pd.concat((pd.read_csv(f, usecols=usecols) for f in filepaths))
read["csv"]         ["pandas"] = lambda filepath,p,filetype:  benchmark(read_csv_pandas, (filepath, Kwargs(usecols=p.columns)), filetype)
read["h5"]          ["pandas"] = lambda filepath,p,filetype:  benchmark(pd.read_hdf, (filepath, p.base, Kwargs(columns=p.columns)), filetype)
read["feather"]     ["pandas"] = lambda filepath,p,filetype:  benchmark(feather.read_dataframe, (filepath,), filetype)
def read_bcolz_pandas(filepath, chunksize=None):
    return bcolz.ctable(rootdir=filepath).todataframe(columns=p.columns)
read["bcolz"]       ["pandas"]   = lambda filepath,p,filetype:  benchmark(read_bcolz_pandas, (filepath, Kwargs(chunksize=1000000)), filetype)
def read_parq_pandas(filepath):
    return fp.ParquetFile(filepath).to_pandas()
read["parq"]        ["pandas"] = lambda filepath,p,filetype:  benchmark(read_parq_pandas, (filepath,), filetype)
read["gz.parq"]     ["pandas"] = lambda filepath,p,filetype:  benchmark(read_parq_pandas, (filepath,), filetype)
read["snappy.parq"] ["pandas"] = lambda filepath,p,filetype:  benchmark(read_parq_pandas, (filepath,), filetype)


write = odict([(f,odict()) for f in ["parq","snappy.parq","gz.parq","bcolz","feather","h5","csv"]])

write["csv"]          ["dask"]   = lambda df,filepath,p:  benchmark(df.to_csv, (filepath.replace(".csv","*.csv"), Kwargs(index=False)))
write["h5"]           ["dask"]   = lambda df,filepath,p:  benchmark(df.to_hdf, (filepath, p.base))
def write_bcolz_dask(filepath, df):
    return bcolz.ctable.fromdataframe(df.compute(), rootdir=filepath)
write["bcolz"]        ["dask"] = lambda df,filepath,p:  benchmark(write_bcolz_dask, (filepath, df))
def write_feather_dask(filepath, df):
    return feather.write_dataframe(df.compute(), filepath)
write["feather"]      ["dask"] = lambda df,filepath,p:  benchmark(write_feather_dask, (filepath, df))
write["parq"]         ["dask"]   = lambda df,filepath,p:  benchmark(dd.to_parquet, (filepath, df)) # **p.parq_opts
write["snappy.parq"]  ["dask"]   = lambda df,filepath,p:  benchmark(dd.to_parquet, (filepath, df, Kwargs(compression='SNAPPY'))) ## **p.parq_opts
write["gz.parq"]      ["dask"]   = lambda df,filepath,p:  benchmark(dd.to_parquet, (filepath, df, Kwargs(compression='GZIP')))

write["csv"]          ["pandas"] = lambda df,filepath,p:  benchmark(df.to_csv, (filepath, Kwargs(index=False)))
write["h5"]           ["pandas"] = lambda df,filepath,p:  benchmark(df.to_hdf, (filepath, Kwargs(key=p.base, format='table')))
write["bcolz"]        ["pandas"] = lambda df,filepath,p:  benchmark(bcolz.ctable.fromdataframe, (df, Kwargs(rootdir=filepath)))
write["feather"]      ["pandas"] = lambda df,filepath,p:  benchmark(feather.write_dataframe, (df, filepath))
write["parq"]         ["pandas"] = lambda df,filepath,p:  benchmark(fp.write, (filepath, df, Kwargs(**p.parq_opts)))
write["gz.parq"]      ["pandas"] = lambda df,filepath,p:  benchmark(fp.write, (filepath, df, Kwargs(compression='GZIP', **p.parq_opts)))
write["snappy.parq"]  ["pandas"] = lambda df,filepath,p:  benchmark(fp.write, (filepath, df, Kwargs(compression='SNAPPY', **p.parq_opts)))


def timed_write(filepath,dftype,fsize='double',output_directory="times"):
    """Accepts any file with a dataframe readable by the given dataframe type, and writes it out as a variety of file types"""
    assert fsize in ('single', 'double')

    p.dftype = dftype # This function may get called from outside main()
    df,duration=timed_read(filepath,dftype)

    for ext in write.keys():
        directory,filename = os.path.split(filepath)
        basename, extension = os.path.splitext(filename)
        fname = output_directory+os.path.sep+basename+"."+ext
        if os.path.exists(fname):
            print("{:28} (keeping existing)".format(fname), flush=True)
        else:
            filetype=ext.split(".")[-1]
            if not filetype in filetypes_storing_categories:
                for c in p.categories:
                    if filetype == 'parq' and df[c].dtype == 'object':
                        df[c]=df[c].str.encode('utf8')
                    else:
                        df[c]=df[c].astype(str)

            # Convert doubles to floats when writing out datasets
            if fsize == 'single':
                for colname in df.columns:
                    if df[colname].dtype == 'float64':
                        df[colname] = df[colname].astype(np.float32)

            code = write[ext].get(dftype,None)

            if code is None:
                print("{:28} {:7} Operation not supported".format(fname,dftype), flush=True)
            else:
                duration, res = code(df,fname,p)
                print("{:28} {:7} {:05.2f}".format(fname,dftype,duration), flush=True)

            if not filetype in filetypes_storing_categories:
                for c in p.categories:
                    df[c]=df[c].astype('category')

        
def timed_read(filepath,dftype):
    basename, extension = os.path.splitext(filepath)
    extension = extension[1:]
    filetype=extension.split(".")[-1]
    code = read[extension].get(dftype,None)

    if code is None:
        return (None, -1)

    p.columns=[p.x]+[p.y]+p.categories
    
    duration, df = code(filepath,p,filetype)
    
    return df, duration


CACHED_RANGES = (None, None)
def timed_agg(df, filepath, plot_width=int(900), plot_height=int(900*7.0/12), cache_ranges=True):
    global CACHED_RANGES
    start = time.time()
    cvs = ds.Canvas(plot_width, plot_height, x_range=CACHED_RANGES[0], y_range=CACHED_RANGES[1])
    agg = cvs.points(df, p.x, p.y)
    end = time.time()
    if cache_ranges:
        CACHED_RANGES = (cvs.x_range, cvs.y_range)
    img = export_image(tf.shade(agg),filepath,export_path=".")
    return img, end-start


def get_size(path):
    total = 0

    # CSV files are broken up by dask when they're written out
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif path.endswith('csv'):
        for csv_fpath in glob.glob(path.replace('.csv', '*.csv')):
            total += os.path.getsize(csv_fpath)
        return total

    # If path is a directory (such as parquet), sum all files in directory
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)

    return total


def get_proc_mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main(argv):
    global DEBUG, DD_FORCE_LOAD, DASK_CLIENT

    parser = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('filepath')
    parser.add_argument('dftype')
    parser.add_argument('base')
    parser.add_argument('x')
    parser.add_argument('y')
    parser.add_argument('categories', nargs='+')
    parser.add_argument('--debug', action='store_true', help='Enable increased verbosity and DEBUG messages')
    parser.add_argument('--cache', choices=('persist', 'cachey'), default=None, help='Enable caching: "persist" causes Dask dataframes to force loading into memory; "cachey" uses dask.cache.Cache with a cachesize of {}. Caching is disabled by default'.format(int(p.cachesize)))
    parser.add_argument('--distributed', action='store_true', help='Enable the distributed scheduler instead of the threaded, which is the default.')
    parser.add_argument('--recalc-ranges', action='store_true', help='Tell datashader to recalculate the ranges on each aggregation, instead of caching them (by default).')
    args = parser.parse_args(argv[1:])

    if args.cache is None:
        if args.debug:
            print("DEBUG: Cache disabled", flush=True)
    else:
        if args.cache == 'cachey':
            from dask.cache import Cache
            cache = Cache(p.cachesize)
            cache.register()
        elif args.cache == 'persist':
            DD_FORCE_LOAD = True

        if args.debug:
            print('DEBUG: Cache "{}" mode enabled'.format(args.cache), flush=True)

    if args.dftype == 'dask' and args.distributed:
        local_cluster = distributed.LocalCluster(n_workers=p.n_workers, threads_per_worker=1)
        DASK_CLIENT = distributed.Client(local_cluster)
        if args.debug:
            print('DEBUG: "distributed" scheduler is enabled')
    else:
        if args.dftype != 'dask' and args.distributed:
            raise ValueError('--distributed argument is only available with the dask dataframe type (not pandas)')
        if args.debug:
            print('DEBUG: "threaded" scheduler is enabled')

    filepath = args.filepath
    basename, extension = os.path.splitext(filepath)
    p.dftype      = args.dftype
    p.base        = args.base
    p.x           = args.x
    p.y           = args.y
    p.categories  = args.categories
    DEBUG = args.debug

    if DEBUG:
        print('DEBUG: Memory usage (before read):\t{} MB'.format(get_proc_mem(), flush=True))
    df,loadtime = timed_read(filepath, p.dftype)

    if df is None:
        if loadtime == -1:
            print("{:28} {:6}  Operation not supported".format(filepath, p.dftype), flush=True)
        return 1

    if DEBUG:
        print('DEBUG: Memory usage (after read):\t{} MB'.format(get_proc_mem(), flush=True))

    img,aggtime1 = timed_agg(df,filepath,5,5,cache_ranges=(not args.recalc_ranges))
    if DEBUG:
        mem_usage = df.memory_usage(deep=True)
        if p.dftype == 'dask':
            mem_usage = mem_usage.compute()
        print('DEBUG:', mem_usage, flush=True)
        mem_usage_total = mem_usage.sum()
        print('DEBUG: DataFrame size:\t\t\t{} MB'.format(mem_usage_total / 1e6, flush=True))
        for colname in df.columns:
            print('DEBUG: column "{}" dtype: {}'.format(colname, df[colname].dtype))
        print('DEBUG: Memory usage (after agg1):\t{} MB'.format(get_proc_mem(), flush=True))

    img,aggtime2 = timed_agg(df,filepath,cache_ranges=(not args.recalc_ranges))
    if DEBUG:
        print('DEBUG: Memory usage (after agg2):\t{} MB'.format(get_proc_mem(), flush=True))
    
    in_size  = get_size(filepath)
    out_size = get_size(filepath+".png")
    
    global_end = time.time()
    print("{:28} {:6}  Aggregate1:{:06.2f} ({:06.2f}+{:06.2f})  Aggregate2:{:06.2f}  In:{:011d}  Out:{:011d}  Total:{:06.2f}"\
          .format(filepath, p.dftype, loadtime+aggtime1, loadtime, aggtime1, aggtime2, in_size, out_size, global_end-global_start), flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
