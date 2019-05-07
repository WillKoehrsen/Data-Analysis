from __future__ import division

if __name__ == "__main__":
    from bokeh.io import curdoc
    from bokeh.plotting import Figure
    from bokeh.models import ColumnDataSource, CustomJS
    from bokeh.tile_providers import STAMEN_TONER
    
    import rasterio as rio
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.colors import Hot
    
    def on_dims_change(attr, old, new):
        update_image()
    
    def update_image():
    
        global dims, raster_data
    
        dims_data = dims.data
    
        if not dims_data['width'] or not dims_data['height']:
            return
    
        xmin = max(dims_data['xmin'][0], raster_data.bounds.left)
        ymin = max(dims_data['ymin'][0], raster_data.bounds.bottom)
        xmax = min(dims_data['xmax'][0], raster_data.bounds.right)
        ymax = min(dims_data['ymax'][0], raster_data.bounds.top)
    
        canvas = ds.Canvas(plot_width=dims_data['width'][0],
                           plot_height=dims_data['height'][0],
                           x_range=(xmin, xmax),
                           y_range=(ymin, ymax))
    
        agg = canvas.raster(raster_data)
        img = tf.shade(agg, cmap=Hot, how='linear')
    
        new_data = {}
        new_data['image'] = [img.data]
        new_data['x'] = [xmin]
        new_data['y'] = [ymin]
        new_data['dh'] = [ymax - ymin]
        new_data['dw'] = [xmax - xmin]
        image_source.stream(new_data, 1)
    
    # load nyc taxi data
    path = './data/projected.tif'
    raster_data = rio.open(path)
    
    # manage client-side dimensions
    dims = ColumnDataSource(data=dict(width=[], height=[], xmin=[], xmax=[], ymin=[], ymax=[]))
    dims.on_change('data', on_dims_change)
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
    
    path = './data/projected.tif'
    
    fig = Figure(x_range=(xmin, xmax),
                 y_range=(ymin, ymax),
                 plot_height=600,
                 plot_width=900,
                 tools='pan,wheel_zoom')
    fig.background_fill_color = 'black'
    fig.add_tile(STAMEN_TONER, alpha=0) # used to set axis ranges
    fig.x_range.callback = CustomJS(code=dims_jscode, args=dict(plot=fig, dims=dims))
    fig.y_range.callback = CustomJS(code=dims_jscode, args=dict(plot=fig, dims=dims))
    fig.axis.visible = False
    fig.grid.grid_line_alpha = 0
    fig.min_border_left = 0
    fig.min_border_right = 0
    fig.min_border_top = 0
    fig.min_border_bottom = 0
    
    image_source = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    fig.image_rgba(source=image_source,
                   image='image',
                   x='x',
                   y='y',
                   dw='dw',
                   dh='dh',
                   dilate=False)
    
    curdoc().add_root(fig)
