"""
A script to generate the actual unemployment vs. forecasts plot up to the
latest available value.

"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fig_1(dfs, params, output_path):
    # Unpack arguments
    (df_unemp_monthly,
     _,
     df_forecast_quarterly,
     df_forecast_monthly,
     _,
     _) = dfs
    analysis_date = params['analysis_date']
    factor = params['factor']

    # Figure 1
    data_start = '2020-03'
    last_date = df_unemp_monthly.index[-1]

    fig = go.Figure()

    # Plot actual unemployment
    fig.add_trace(go.Scatter(x=df_forecast_monthly[data_start:].index,
                             y=df_forecast_monthly.post_covid[data_start:],
                             name='Actual Unemployment',
                            marker=dict(color='green', size=8)))

    # Plot pre-covid projections
    x_Q = df_forecast_quarterly[data_start:].index.shift(periods=31, freq='D')
    y_Q = df_forecast_quarterly.pre_covid[data_start:]
    fig.add_trace(go.Scatter(x=x_Q,
                             y=y_Q,
                             mode='markers',
                             showlegend=False,
                             marker=dict(color='blue', size=8)))

    fig.add_trace(go.Scatter(x=df_forecast_monthly[data_start:].index,
                             y=df_forecast_monthly.pre_covid[data_start:],
                             name='Pre-Covid Projections',
                             marker=dict(color='blue')))

    # Add dashed line at last available date
    fig.update_layout(yaxis_title='Unemployment Rate',
                      xaxis_range=[data_start, last_date])

    write_path = output_path + analysis_date + '/fig_1_unemp_present.pdf'
    fig.write_image(write_path, format='pdf')


if __name__ == '__main__':
    import yaml
    from load_data import load_data

    params_path = '../parameters.yml'
    output_path = '../output/'
    clean_data_path = '../clean_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    dfs = load_data(params, clean_data_path)
    fig_1(dfs, params, output_path)
