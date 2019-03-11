from __future__ import division

import math

from collections import OrderedDict

from bokeh.io import curdoc
from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.tile_providers import STAMEN_TONER
from bokeh.models import VBox, HBox, Paragraph, Select
from bokeh.palettes import BuGn9

import pandas as pd

import datashader as ds
import datashader.transfer_functions as tf

def bin_data():
    global time_period, grouped, group_count, counter, times, groups
    grouped = df.groupby([times.hour, times.minute // time_period])
    groups = sorted(grouped.groups.keys(), key=lambda r: (r[0], r[1]))
    group_count = len(groups)
    counter = 0

def on_time_select_change(attr, old, new):
    global time_period, counter, time_select_options
    time_period = time_select_options[new]
    counter = 0
    bin_data()

counter = 0
def update_data():
    global dims, grouped, group_count, counter, time_text, time_period

    dims_data = dims.data

    if not dims_data['width'] or not dims_data['height']:
        return

    group_num = counter % group_count
    group = groups[group_num]
    grouped_df = grouped.get_group(group)
    update_image(grouped_df)

    # update time text
    num_minute_groups = 60 // time_period
    mins = group[1] * time_period
    hr = group[0]
    end_mins = ((group[1] + 1) % num_minute_groups) * time_period
    end_hr = hr if end_mins > 0 else (hr + 1) % 24
    time_text.text = 'Time Period: {}:{} - {}:{}'.format(str(hr).zfill(2),
                                                         str(mins).zfill(2),
                                                         str(end_hr).zfill(2),
                                                         str(end_mins).zfill(2))
    counter += 1

def update_image(dataframe):
    global dims
    dims_data = dims.data

    if not dims_data['width'] or not dims_data['height']:
        return

    plot_width = int(math.ceil(dims_data['width'][0]))
    plot_height = int(math.ceil(dims_data['height'][0]))
    x_range = (dims_data['xmin'][0], dims_data['xmax'][0])
    y_range = (dims_data['ymin'][0], dims_data['ymax'][0])

    canvas = ds.Canvas(plot_width=plot_width,
                       plot_height=plot_height,
                       x_range=x_range,
                       y_range=y_range)

    agg = canvas.points(dataframe, 'dropoff_x', 'dropoff_y',
                        ds.count('trip_distance'))

    img = tf.shade(agg, cmap=BuGn9, how='log')

    new_data = {}
    new_data['image'] = [img.data]
    new_data['x'] = [x_range[0]]
    new_data['y'] = [y_range[0]]
    new_data['dh'] = [y_range[1] - y_range[0]]
    new_data['dw'] = [x_range[1] - x_range[0]]

    image_source.stream(new_data, 1)


time_select_options = OrderedDict()
time_select_options['1 Hour'] = 60
time_select_options['30 Minutes'] = 30
time_select_options['15 Minutes'] = 15
time_period = list(time_select_options.values())[0]

time_select = Select.create(name="Time Period", options=time_select_options)
time_select.on_change('value', on_time_select_change)

time_text = Paragraph(text='Time Period')

# load nyc taxi data
path = './data/nyc_taxi.csv'
datetime_field = 'tpep_dropoff_datetime'
cols = ['dropoff_x', 'dropoff_y', 'trip_distance', datetime_field]

df = pd.read_csv(path, usecols=cols, parse_dates=[datetime_field]).dropna(axis=0)
times = pd.DatetimeIndex(df[datetime_field])
group_count = grouped = groups = None
bin_data()

# manage client-side dimensions
dims = ColumnDataSource(data=dict(width=[], height=[], xmin=[], xmax=[], ymin=[], ymax=[]))
dims_jscode = """
var update_dims = function () {
    var new_data = {
        height: [plot.frame.height],
        width: [plot.frame.width],
        xmin: [plot.x_range.start],
        ymin: [plot.y_range.start],
        xmax: [plot.x_range.end],
        ymax: [plot.y_range.end]
    };
    dims.data = new_data;
};

if (typeof throttle != 'undefined' && throttle != null) {
    clearTimeout(throttle);
}

throttle = setTimeout(update_dims, 100, "replace");
"""

# Create plot -------------------------------
xmin = -8240227.037
ymin = 4974203.152
xmax = -8231283.905
ymax = 4979238.441

fig = Figure(x_range=(xmin, xmax),
             y_range=(ymin, ymax),
             plot_height=600,
             plot_width=900,
             tools='pan,wheel_zoom')
fig.background_fill_color = 'black'
fig.add_tile(STAMEN_TONER, alpha=.3)
fig.x_range.callback = CustomJS(code=dims_jscode, args=dict(plot=fig, dims=dims))
fig.y_range.callback = CustomJS(code=dims_jscode, args=dict(plot=fig, dims=dims))
fig.axis.visible = False
fig.grid.grid_line_alpha = 0
fig.min_border_left = 0
fig.min_border_right = 0
fig.min_border_top = 0
fig.min_border_bottom = 0

image_source = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
fig.image_rgba(source=image_source, image='image', x='x', y='y', dw='dw', dh='dh', dilate=False)

time_text = Paragraph(text='Time Period: 00:00 - 00:00')
controls = HBox(children=[time_text, time_select], width=fig.plot_width)
layout = VBox(children=[fig, controls])

curdoc().add_root(layout)
curdoc().add_periodic_callback(update_data, 1000)
