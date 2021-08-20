"""
A module containing routines for creating plots of the model's output.

"""

import calendar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .model import compute_induced_suicides


COVID_START = '2020-03'


def plot_unemp_suicide(suicide, unemp):
    color = suicide.index.year + (suicide.index.month - 1) / 12

    fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {}]])

    x_vals = unemp.index.intersection(suicide.index)

    fig.add_trace(go.Scatter(x=x_vals,
                             y=unemp,
                             name=unemp.name + ' Unemployment Rate (Left)',
                             marker=dict(color='blue')),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=x_vals,
                             y=suicide,
                             name=suicide.name + ' Number of Suicides (Right)',
                             marker=dict(color='red')),
                  secondary_y=True,
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=unemp,
                             y=suicide,
                             mode='markers',
                             marker=dict(color=color,
                                         colorbar=dict(thickness=10)),
                             showlegend=False),
                  row=1,
                  col=2)

    # Update figure size
    fig.update_layout(height=600, width=1200)

    # Update xaxis properties
    fig.update_xaxes(title_text=unemp.name + " Unemployment Rate",
                     row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text=suicide.name + " Number of Suicides",
                     row=1, col=2)

    # Update legend
    legend = dict(yanchor="top", y=1.15, xanchor="left", x=0.01)
    fig.update_layout(legend=legend)

    return fig


def plot_unemployment(forecasts, last_date):
    # Create figure
    fig = go.Figure()

    # Plot actual unemployment
    monthly_data = forecasts[COVID_START:last_date]

    fig.add_trace(go.Scatter(x=monthly_data.index,
                             y=monthly_data.post_covid,
                             name='Actual Unemployment',
                             marker=dict(color='green', size=8)))

    # Plot pre-covid projections
    fig.add_trace(go.Scatter(x=monthly_data.index,
                             y=monthly_data.pre_covid,
                             name='Pre-Covid Projections',
                             marker=dict(color='blue')))

    # Add dashed line at last available date
    fig.update_layout(yaxis_title='Unemployment Rate')

    return fig


def plot_forecasts(forecasts, last_date):

    fig = go.Figure()

    # Plot actual unemployment
    monthly_data = forecasts[COVID_START:].copy()

    # Plot actual unemployment
    fig.add_trace(go.Scatter(x=monthly_data.index,
                             y=monthly_data.post_covid,
                             name='Actual Unemployment',
                             marker=dict(color='green', size=8)))

    # Plot post-covid projections
    fig.add_trace(go.Scatter(x=monthly_data[last_date:].index,
                             y=monthly_data[last_date:].post_covid,
                             mode='lines',
                             name='Post-Covid Projections'))

    fig.add_trace(go.Scatter(x=monthly_data.index,
                             y=monthly_data.pre_covid,
                             name='Pre-Covid Projections',
                             marker=dict(color='blue')))

    # Add dashed line at last available date
    fig.add_vline(x=last_date, line_dash='dash')

    fig.update_layout(yaxis_title='Unemployment Rate')

    return fig


def plot_preds_by_month(pre_preds, post_preds, suicide):

    # Manual groupby for better plotting
    month_inds = suicide.index.month.unique().sort_values()
    month_names = [calendar.month_name[val] for val in month_inds]
    fig = make_subplots(4, 3, subplot_titles=month_names)

    # Add traces for each month
    for month_idx in month_inds:
        # Hash to determine row and col indices
        row_idx = (month_idx - 1) // 3 + 1
        col_idx = (month_idx - 1) % 3 + 1
        showlegend = True if (row_idx == 1) & (col_idx == 1) else False

        suicide_month = suicide[suicide.index.month == month_idx]
        pre_month = pre_preds[pre_preds.index.month == month_idx]
        post_month = post_preds[post_preds.index.month == month_idx]

        # Actual suicides
        fig.add_trace(go.Scatter(x=suicide_month.index,
                                 y=suicide_month,
                                 name='Observed',
                                 mode='markers',
                                 marker=dict(color='red'),
                                 showlegend=showlegend),
                      row=row_idx,
                      col=col_idx)

        # Pre-covid predictions
        fig.add_trace(go.Scatter(x=pre_month.index,
                                 y=pre_month,
                                 name='Expected (Pre-Covid)',
                                 marker=dict(color='blue'),
                                 showlegend=showlegend),
                      row=row_idx,
                      col=col_idx)

        # Post-covid predictions
        fig.add_trace(go.Scatter(x=post_month.index,
                                 y=post_month,
                                 name='Expected (Post-Covid)',
                                 marker=dict(color='green'),
                                 showlegend=showlegend),
                      row=row_idx,
                      col=col_idx)

    # Set x-axis title
    for i in range(1, 13):
        fig['layout']['xaxis{}'.format(i)]['title'] = 'Year'

    # Update size and title
    temp = ' Observed Versus Expected Number of Suicides by Month Over Time'
    title = suicide.name + temp
    fig.update_layout(height=1200, width=900, title=title)

    return fig


def plot_preds_ts(pre_preds, post_preds, suicide, pre_conf_int):
    fig = go.Figure()

    # Confidence intervals
    fig.add_trace(go.Scatter(x=pre_preds.index,
                             y=pre_conf_int[:, 0],  # Lower
                             fillcolor='rgba(0,0,255,0.1)',
                             line=dict(color='rgba(0,0,255,0.)'),
                             hoverinfo="skip",
                             showlegend=False))

    fig.add_trace(go.Scatter(x=pre_preds.index,
                             y=pre_conf_int[:, 1],  # Upper
                             fill='tonexty',
                             fillcolor='rgba(0,0,255,0.1)',
                             line=dict(color='rgba(0,0,255,0.)'),
                             hoverinfo="skip",
                             showlegend=False))

    # Lines
    fig.add_trace(go.Scatter(x=suicide.index,
                             y=suicide,
                             name='Observed',
                             mode='markers',
                             marker=dict(color='red')))

    fig.add_trace(go.Scatter(x=pre_preds.index,
                             y=pre_preds,
                             name='Expected (Pre-Covid)',
                             marker=dict(color='blue')))

    fig.add_trace(go.Scatter(x=pre_preds.index,
                             y=post_preds,
                             name='Expected (Post-Covid)',
                             marker=dict(color='green')))

    # Add line for start of the covid-19 pandemic
    line = '2020-02-15'
    fig.add_vline(x=line,
                  line_width=2,
                  line_dash="dash",
                  line_color="black",
                  name='Pandemic start')

    # Update title
    temp = ' Observed Versus Expected Number of Suicides'
    title = suicide.name + temp
    fig.update_layout(title=title)

    return fig


def plot_induced_suicides_ts(pre_preds, suicide):
    data = compute_induced_suicides(suicide, pre_preds)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=data.index,
                         y=data,
                         showlegend=False))

    temp = ' Covid-induced Suicides Over Time'
    title = suicide.name + temp
    fig.update_layout(title=title)

    return fig


def plot_explained_unemp_ts(pre_preds, post_preds, name):

    data = post_preds[COVID_START:] - pre_preds[COVID_START:]

    fig = go.Figure()

    fig.add_trace(go.Bar(x=data.index,
                         y=data,
                         showlegend=False))

    temp = ' Number of Suicides Explained by Unemployment Over Time'
    title = name + temp
    fig.update_layout(title=title)

    return fig
