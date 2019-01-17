import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from scipy import stats

import plotly.graph_objs as go
import cufflinks
cufflinks.go_offline()


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
        xaxis=dict(title=x.replace('_', ' ').title()),
        title=f"{x.replace('_', ' ').title()} Distribution by {category.replace('_', ' ').title()}"
        if category
        else f"{x.replace('_', ' ').title()} Distribution",
    )

    figure = go.Figure(data=data, layout=layout)
    return figure


def make_cum_plot(df, y, category=None, ranges=False):
    """
    Make an interactive cumulative plot, optionally segmented by `category`

    :param df: dataframe of data, must have a `published_date` column
    :param y: string of column to use for plotting or list of two strings for double y axis
    :param category: string representing column to segment by
    :param ranges: boolean for whether to add range slider and range selector

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
                    marker=dict(size=10, opacity=0.8,
                                symbol=i + 2),
                )
            )
    else:
        df.sort_values("published_date", inplace=True)
        if len(y) == 2:
            data = [
                go.Scatter(
                    x=df["published_date"],
                    y=df[y[0]].cumsum(),
                    name=y[0].title(),
                    mode="lines+markers",
                    text=df["title"],
                    marker=dict(size=10, color='blue', opacity=0.6, line=dict(color='black'),
                                )),
                go.Scatter(
                    x=df["published_date"],
                    y=df[y[1]].cumsum(),
                    yaxis='y2',
                    name=y[1].title(),
                    mode="lines+markers",
                    text=df["title"],
                    marker=dict(size=10, color='red', opacity=0.6, line=dict(color='black'),
                                )),
            ]
        else:
            data = [
                go.Scatter(
                    x=df["published_date"],
                    y=df[y].cumsum(),
                    mode="lines+markers",
                    text=df["title"],
                    marker=dict(size=12, color='blue', opacity=0.6, line=dict(color='black'),
                                ),
                )
            ]
    if len(y) == 2:
        layout = go.Layout(
            xaxis=dict(title="Published Date", type="date"),
            yaxis=dict(title=y[0].replace('_', ' ').title(), color='blue'),
            yaxis2=dict(title=y[1].replace('_', ' ').title(), color='red',
                        overlaying='y', side='right'),
            font=dict(size=14),
            title=f"Cumulative {y[0].title()} and {y[1].title()}",
        )
    else:
        layout = go.Layout(
            xaxis=dict(title="Published Date", type="date"),
            yaxis=dict(title=y.replace('_', ' ').title()),
            font=dict(size=14),
            title=f"Cumulative {y.replace('_', ' ').title()} by {category.replace('_', ' ').title()}"
            if category is not None
            else f"Cumulative {y.replace('_', ' ').title()}",
        )

    # Add a rangeselector and rangeslider for a data xaxis
    if ranges:
        rangeselector = dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        )
        rangeslider = dict(visible=True)
        layout["xaxis"]["rangeselector"] = rangeselector
        layout["xaxis"]["rangeslider"] = rangeslider
        layout['width'] = 1000
        layout['height'] = 600

    figure = go.Figure(data=data, layout=layout)
    return figure


def make_scatter_plot(df, x, y, fits=None, xlog=False, ylog=False, category=None, scale=None, sizeref=2, annotations=None, ranges=False, title_override=None):
    """
    Make an interactive scatterplot, optionally segmented by `category`

    :param df: dataframe of data
    :param x: string of column to use for xaxis
    :param y: string of column to use for yaxis
    :param fits: list of strings of fits
    :param xlog: boolean for making a log xaxis
    :param ylog boolean for making a log yaxis
    :param category: string representing categorical column to segment by, this must be a categorical
    :param scale: string representing numerical column to size and color markers by, this must be numerical data
    :param sizeref: float or integer for setting the size of markers according to the scale, only used if scale is set
    :param annotations: text to display on the plot (dictionary)
    :param ranges: boolean for whether to add a range slider and selector
    :param title_override: String to override the title

    :return figure: a plotly plot to show with iplot or plot
    """
    if category is not None:
        title = f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()} by {category.replace('_', ' ').title()}"
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
            title = f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()} Scaled by {scale.title()}"
            data = [go.Scatter(x=df[x],
                               y=df[y],
                               mode='markers',
                               text=df['title'], marker=dict(size=df[scale],
                                                             line=dict(color='black', width=0.5), sizemode='area', sizeref=sizeref, opacity=0.8,
                                                             colorscale='Viridis', color=df[scale], showscale=True, sizemin=2))]
        else:

            df.sort_values(x, inplace=True)
            title = f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}"
            data = [go.Scatter(x=df[x],
                               y=df[y],
                               mode='markers',
                               text=df['title'], marker=dict(
                size=12, color='blue', opacity=0.8, line=dict(color='black')),
                name='observations')]
            if fits is not None:
                for fit in fits:
                    data.append(go.Scatter(x=df[x], y=df[fit], text=df['title'],
                                           mode='lines+markers', marker=dict
                                           (size=8, opacity=0.6),
                                           line=dict(dash='dash'), name=fit))

                title += ' with Fit'
    layout = go.Layout(annotations=annotations,
                       xaxis=dict(title=x.replace('_', ' ').title() + (' (log scale)' if xlog else ''),
                                  type='log' if xlog else None),
                       yaxis=dict(title=y.replace('_', ' ').title() + (' (log scale)' if ylog else ''),
                                  type='log' if ylog else None),
                       font=dict(size=14),
                       title=title if title_override is None else title_override,
                       )

    # Add a rangeselector and rangeslider for a data xaxis
    if ranges:
        rangeselector = dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        )
        rangeslider = dict(visible=True)
        layout["xaxis"]["rangeselector"] = rangeselector
        layout["xaxis"]["rangeslider"] = rangeslider
        layout['width'] = 1000
        layout['height'] = 600

    figure = go.Figure(data=data, layout=layout)
    return figure


def make_linear_regression(df, x, y, intercept_0):
    """
    Create a linear regression, either with the intercept set to 0 or
    the intercept allowed to be fitted

    :param df: dataframe with data
    :param x: string or list of stringsfor the name of the column with x data
    :param y: string for the name of the column with y data
    :param intercept_0: boolean indicating whether to set the intercept to 0
    """
    if isinstance(x, list):
        lin_model = LinearRegression()
        lin_model.fit(df[x], df[y])

        slopes, intercept, = lin_model.coef_, lin_model.intercept_
        df['predicted'] = lin_model.predict(df[x])
        r2 = lin_model.score(df[x], df[y])
        rmse = np.sqrt(mean_squared_error(
            y_true=df[y], y_pred=df['predicted']))
        equation = f'{y.replace("_", " ")} ='

        names = ['r2', 'rmse', 'intercept']
        values = [r2, rmse, intercept]
        for i, (p, s) in enumerate(zip(x, slopes)):
            if (i + 1) % 3 == 0:
                equation += f'<br>{s:.2f} * {p.replace("_", " ")} +'
            else:
                equation += f' {s:.2f} * {p.replace("_", " ")} +'
            names.append(p)
            values.append(s)

        equation += f' {intercept:.2f}'
        annotations = [dict(x=0.4 * df.index.max(), y=0.9 * df[y].max(), showarrow=False,
                            text=equation,
                            font=dict(size=10))]

        df['index'] = list(df.index)
        figure = make_scatter_plot(df, x='index', y=y, fits=[
                                   'predicted'], annotations=annotations)
        summary = pd.DataFrame({'name': names, 'value': values})
    else:
        if intercept_0:
            lin_reg = sm.OLS(df[y], df[x]).fit()
            df['fit_values'] = lin_reg.fittedvalues
            summary = lin_reg.summary()
            slope = float(lin_reg.params)
            equation = f"${y.replace('_', ' ')} = {slope:.2f} * {x.replace('_', ' ')}$"

        else:
            lin_reg = stats.linregress(df[x], df[y])
            intercept, slope = lin_reg.intercept, lin_reg.slope
            params = ['pvalue', 'rvalue', 'slope', 'intercept']
            values = []
            for p in params:
                values.append(getattr(lin_reg, p))
            summary = pd.DataFrame({'param': params, 'value': values})
            df['fit_values'] = df[x] * slope + intercept
            equation = f"${y.replace('_', ' ')} = {slope:.2f} * {x.replace('_', ' ')} + {intercept:.2f}$"

        annotations = [dict(x=0.75 * df[x].max(), y=0.9 * df[y].max(), showarrow=False,
                            text=equation,
                            font=dict(size=32))]
        figure = make_scatter_plot(
            df, x=x, y=y, fits=['fit_values'], annotations=annotations)
    return figure, summary


def make_poly_fits(df, x, y, degree=6):
    """
    Generate fits and make interactive plot with fits

    :param df: dataframe with data
    :param x: string representing x data column
    :param y: string representing y data column
    :param degree: integer degree of fits to go up to

    :return fit_stats: dataframe with information about fits
    :return figure: interactive plotly figure that can be shown with iplot or plot
    """

    # Don't want to alter original data frame
    df = df.copy()
    fit_list = []
    rmse = []
    fit_params = []

    # Make each fit
    for i in range(1, degree + 1):
        fit_name = f'fit degree = {i}'
        fit_list.append(fit_name)
        z, res, *rest = np.polyfit(df[x], df[y], i, full=True)
        fit_params.append(z)
        df.loc[:, fit_name] = np.poly1d(z)(df[x])
        rmse.append(np.sqrt(res[0]))

    fit_stats = pd.DataFrame(
        {'fit': fit_list, 'rmse': rmse, 'params': fit_params})
    figure = make_scatter_plot(df, x=x, y=y, fits=fit_list)
    return figure, fit_stats


def make_extrapolation(df, y, years, degree=4):
    """
    Extrapolate `y` into the future `years` with `degree`  polynomial fit

    :param df: dataframe of data
    :param y: string of column to extrapolate
    :param years: number of years to extrapolate into the future
    :param degree: integer degree of polynomial fit

    :return figure: plotly figure for display using iplot or plot
    :return future_df: extrapolated numbers into the future
    """

    df = df.copy()
    x = 'days_since_start'
    df['days_since_start'] = (
        (df['published_date'] - df['published_date'].min()).
        dt.total_seconds() / (3600 * 24)).astype(int)

    cumy = f'cum_{y}'
    df[cumy] = df.sort_values(x)[y].cumsum()

    figure, summary = make_poly_fits(df, x, cumy, degree=degree)

    min_date = df['published_date'].min()
    max_date = df['published_date'].max()

    date_range = pd.date_range(start=min_date,
                               end=max_date + pd.Timedelta(days=int(years * 365)))

    future_df = pd.DataFrame({'date': date_range})

    future_df[x] = (
        (future_df['date'] - future_df['date'].min()).
        dt.total_seconds() / (3600 * 24)).astype(int)

    newcumy = f'cumulative_{y}'

    future_df = future_df.merge(df[[x, cumy]], on=x, how='left').\
        rename(columns={cumy: newcumy})

    z = np.poly1d(summary.iloc[-1]['params'])
    pred_name = f'predicted_{y}'
    future_df[pred_name] = z(future_df[x])
    future_df['title'] = ''

    last_date = future_df.loc[future_df['date'].idxmax()]
    prediction_text = (
        f"On {last_date['date'].date()} the {y} will be {float(last_date[pred_name]):,.0f}.")
    annotations = [dict(x=future_df['date'].quantile(0.4),
                        y=0.8 * future_df[pred_name].max(), text=prediction_text, showarrow=False,
                        font=dict(size=16))]

    title_override = f'{y.replace("_", " ").title()} with Extrapolation {years} Years into the Future'

    figure = make_scatter_plot(future_df, 'date', newcumy, fits=[
                               pred_name], annotations=annotations, ranges=True, title_override=title_override)
    return figure, future_df
