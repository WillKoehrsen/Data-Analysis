# Data science imports
from multiprocessing import Pool
import requests
import re
from bs4 import BeautifulSoup
from itertools import chain
from collections import Counter, defaultdict
from timeit import default_timer as timer
import pandas as pd


from scipy import stats

# Interactive plotting
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks

cufflinks.go_offline()


def make_update_menu(base_title, article_annotations=None, response_annotations=None):
    """
    Make an updatemenu for interative plot

    :param base_title: string for title of plot

    :return updatemenus: a updatemenus object for adding to a layout
    """
    updatemenus = list(
        [
            dict(
                buttons=list(
                    [
                        dict(
                            label="both",
                            method="update",
                            args=[
                                dict(visible=[True, True]),
                                dict(
                                    title=base_title,
                                    annotations=[
                                        article_annotations,
                                        response_annotations,
                                    ],
                                ),
                            ],
                        ),
                        dict(
                            label="articles",
                            method="update",
                            args=[
                                dict(visible=[True, False]),
                                dict(
                                    title="Article " + base_title,
                                    annotations=[article_annotations],
                                ),
                            ],
                        ),
                        dict(
                            label="responses",
                            method="update",
                            args=[
                                dict(visible=[False, True]),
                                dict(
                                    title="Response " + base_title,
                                    annotations=[response_annotations],
                                ),
                            ],
                        ),
                    ]
                )
            )
        ]
    )
    return updatemenus


def make_hist(df, x, category=None):
    """
    Make an interactive histogram, optionally segmented by `category`

    :param df: dataframe of data
    :param x: string of column to use for plotting
    :param category: string representing column to segment by

    :return figure: a plotly histogram to show with iplot or plot
    """
    if category is not None:
        data = []
        for name, group in df.groupby(category):
            data.append(go.Histogram(dict(x=group[x], name=name)))
    else:
        data = [go.Histogram(dict(x=df[x]))]

    layout = go.Layout(
        yaxis=dict(title="Count"),
        xaxis=dict(title=x.title()),
        title=f"{x.title()} Distribution by {category.title()}"
        if category
        else f"{x.title()} Distribution",
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def make_cum_plot(df, y, category=None):
    """
    Make an interactive cumulative plot, optionally segmented by `category`

    :param df: dataframe of data, must have a `published_date` column
    :param y: string of column to use for plotting
    :param category: string representing column to segment by

    :return figure: a plotly plot to show with iplot or plot
    """
    if category is not None:
        data = []
        for i, (name, group) in enumerate(df.groupby(category)):
            group.sort_values("published_date", inplace=True)
            data.append(
                go.Scatter(
                    x=group["published_date"],
                    y=group[y].cumsum(),
                    mode="lines+markers",
                    text=group["title"],
                    name=name,
                    marker=dict(size=8, symbol=i + 302),
                )
            )
    else:
        df.sort_values("published_date", inplace=True)
        data = [
            go.Scatter(
                x=df["published_date"],
                y=df[y].cumsum(),
                mode="lines+markers",
                text=df["title"],
                marker=dict(size=10),
            )
        ]

    layout = go.Layout(
        xaxis=dict(title="Published Date", type="date"),
        yaxis=dict(title=y.title()),
        font=dict(size=14),
        title=f"Cumulative {y.title()} by {category.title()}"
        if category is not None
        else f"Cumulative {y.title()}",
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def make_scatter_plot(df, x, y, xlog=False, ylog=False, category=None, scale=None):
    """
    Make an interactive scatterplot, optionally segmented by `category`

    :param df: dataframe of data
    :param x: string of column to use for xaxis
    :param y: string of column to use for yaxis
    :param xlog: boolean for making a log xaxis
    :param ylog boolean for making a log yaxis
    :param category: string representing categorical column to segment by, this must be a categorical
    :param scale: string representing numerical column to size and color markers by, this must be numerical data

    :return figure: a plotly plot to show with iplot or plot
    """
    if category is not None:
        title = f"{y.title()} vs {x.title()} by {category.title()}"
        data = []
        for i, (name, group) in enumerate(df.groupby(category)):
            data.append(go.Scatter(x=group[x],
                                   y=group[y],
                                   mode='markers',
                                   text=group['title'],
                                   name=name,
                                   marker=dict(size=8, symbol=i + 2)))

    else:
        if scale is not None:
            title = f"{y.title()} vs {x.title()} by {scale.title()}"
            data = [go.Scatter(x=df[x],
                               y=df[y],
                               mode='markers',
                               text=df['title'], marker=dict(size=df[scale], sizemode='area',
                                                             colorscale='Viridis', color=df[scale], showscale=True, sizemin=2))]
        else:
            title = f"{y.title()} vs {x.title()}"
            data = [go.Scatter(x=df[x],
                               y=df[y],
                               mode='markers',
                               text=df['title'], marker=dict(size=10))]

    layout = go.Layout(
        xaxis=dict(title=x.title() + (' (log scale)' if xlog else ''),
                   type='log' if xlog else None),
        yaxis=dict(title=y.title() + (' (log scale)' if ylog else ''),
                   type='log' if ylog else None),
        font=dict(size=14),
        title=title,
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


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
    responses = data[data["response"] == "response"].copy()
    articles = data[data["response"] == "article"].copy()

    if not responses.empty:
        # Create scatterplot data, articles must be first for menu selection
        plot_data = [
            go.Scatter(
                x=articles[x],
                y=articles[y],
                mode="markers",
                name="articles",
                text=articles["title"],
                marker=dict(color="blue", size=12),
            ),
            go.Scatter(
                x=responses[x],
                y=responses[y],
                mode="markers",
                name="responses",
                marker=dict(color="green", size=12),
            ),
        ]

        if not time:
            annotations = {}
            for df, name in zip([articles, responses], ["articles", "responses"]):

                regression = stats.linregress(x=df[x], y=df[y])
                slope = regression.slope
                intercept = regression.intercept
                rvalue = regression.rvalue

                xi = np.array(range(int(df[x].min()), int(df[x].max())))

                line = xi * slope + intercept
                trace = go.Scatter(
                    x=xi,
                    y=line,
                    mode="lines",
                    marker=dict(color="blue" if name ==
                                "articles" else "green"),
                    line=dict(width=4, dash="longdash"),
                    name=f"{name} linear fit",
                )

                annotations[name] = dict(
                    x=max(xi) * eq_pos[0],
                    y=df[y].max() * eq_pos[1],
                    showarrow=False,
                    text=f"$R^2 = {rvalue:.2f}; Y = {slope:.2f}X + {intercept:.2f}$",
                    font=dict(size=16, color="blue" if name ==
                              "articles" else "green"),
                )

                plot_data.append(trace)

        # Make a layout with update menus
        layout = go.Layout(
            annotations=list(annotations.values()),
            height=600,
            width=900,
            title=base_title,
            xaxis=dict(
                title=x.title(), tickfont=dict(size=14), titlefont=dict(size=16)
            ),
            yaxis=dict(
                title=y.title(), tickfont=dict(size=14), titlefont=dict(size=16)
            ),
            updatemenus=make_update_menu(
                base_title, annotations["articles"], annotations["responses"]
            ),
        )

    # If there are only articles
    else:
        plot_data = [
            go.Scatter(
                x=data[x],
                y=data[y],
                mode="markers",
                name="observations",
                text=data["title"],
                marker=dict(color="blue", size=12),
            )
        ]

        regression = stats.linregress(x=data[x], y=data[y])
        slope = regression.slope
        intercept = regression.intercept
        rvalue = regression.rvalue

        xi = np.array(range(int(data[x].min()), int(data[x].max())))
        line = xi * slope + intercept
        trace = go.Scatter(
            x=xi,
            y=line,
            mode="lines",
            marker=dict(color="red"),
            line=dict(width=4, dash="longdash"),
            name="linear fit",
        )

        annotations = [
            dict(
                x=max(xi) * eq_pos[0],
                y=data[y].max() * eq_pos[1],
                showarrow=False,
                text=f"$R^2 = {rvalue:.2f}; Y = {slope:.2f}X + {intercept:.2f}$",
                font=dict(size=16),
            )
        ]

        plot_data.append(trace)

        layout = go.Layout(
            annotations=annotations,
            height=600,
            width=900,
            title=base_title,
            xaxis=dict(
                title=x.title(), tickfont=dict(size=14), titlefont=dict(size=16)
            ),
            yaxis=dict(
                title=y.title(), tickfont=dict(size=14), titlefont=dict(size=16)
            ),
        )

    # Add a rangeselector and rangeslider for a data xaxis
    if time:
        rangeselector = dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        )
        rangeslider = dict(visible=True)
        layout["xaxis"]["rangeselector"] = rangeselector
        layout["xaxis"]["rangeslider"] = rangeslider

        figure = go.Figure(data=plot_data, layout=layout)

        return figure

    # Return the figure
    figure = go.Figure(data=plot_data, layout=layout)

    return figure
