# Datashader Examples

The best way to understand how Datashader works is to try out our
extensive set of examples. [Datashader.org](http://datashader.org)
includes static versions of the 
[getting started guide](http://datashader.org/getting-started), 
[user manual](http://datashader.org/user-guide), and
[topic examples](http://datashader.org/topics), but for the full
experience with dynamic updating you will need to install them on a
live server. 

These instructions assume you are using 
[conda](https://conda.io/docs/install/quick.html), but they can be 
adapted as needed to use [pip](https://pip.pypa.io/en/stable/installing/) 
and [virtualenv](https://virtualenv.pypa.io) if desired.

To get started, first go to your home directory and
download the current list of everything needed for the examples:

- Download the [conda ds environment file](https://raw.githubusercontent.com/bokeh/datashader/master/examples/environment.yml) and save it as `environment.yml`.

Then run the following commands in your terminal (command) prompt, from wherever you saved `environment.yml`:

```bash
1. conda env create --file environment.yml
2. conda activate ds
3. datashader examples
3. cd datashader-examples
```

Step 1 will read `environment.yml`, create a new Conda environment
named `ds`, and install of the libraries needed into that environment
(including datashader itself). It will use Python 3.6 by default, but
you can edit that file to specify a different Python version if you
prefer (which may require changing some of the dependencies in some
cases).

Step 2 will activate the `ds` environment, using it for all subsequent
commands. You will need to re-run step 2 after closing your terminal or
rebooting your machine, if you want to use anything in the `ds` environment.
For older versions of conda, you may instead need to do `source activate ds`
(mac/linux) or `activate ds` (windows).

Step 3 will copy the datashader examples from wherever Conda placed
them into a subdirectory `datashader-examples`, and will then download
the sample data required for the examples.  (`datashader examples` is
a shorthand for `datashader copy-examples --path datashader-examples
&& datashader fetch-data --path datashader-examples`.)

The total download size is currently about 4GB to transfer, requiring
about 10GB on disk when unpacked, which can take some time depending on
the speed of your connection.  The files involved are specified in the
text file `datasets.yml` in the `datashader-examples` directory, and
you are welcome to edit that file or to download the individual files
specified therein manually if you prefer, as long as you put them into
a subdirectory `data/` so the examples can find them.  Once these
steps have completed, you will be ready to run any of the examples
listed on [datashader.org](http://datashader.org).


## Notebooks

Most of the examples are in the form of runnable Jupyter
notebooks. Once you have obtained the notebooks and the data they
require, you can run them on your own system using Jupyter:

```
cd datashader-examples
jupyter notebook
```

If you want the generated notebooks to work without an internet connection or
with an unreliable connection (e.g. if you see `Loading BokehJS ...` but never
`BokehJS sucessfully loaded`), then restart the Jupyter notebook server using:

```
BOKEH_RESOURCES=inline jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000
```

See dashboard.ipynb in this directory for a Datashder dashboard for viewing data.
