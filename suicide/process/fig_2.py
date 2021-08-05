"""
A script to generate the full actual unemployment vs. forecasts plot.

"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fig_2(dfs, params, output_path):
    # Unpack arguments
    (df_unemp_monthly,
     _,
     df_forecast_quarterly,
     df_forecast_monthly,
     _,
     _) = dfs
    analysis_date = params['analysis_date']
    factor = params['factor']

    # Figure 2
    data_start = '2020-03'
    last_date = df_unemp_monthly.index[-1]

    fig = go.Figure()

    # Plot actual unemployment
    fig.add_trace(go.Scatter(x=df_forecast_monthly[data_start:last_date].index,
                             y=df_forecast_monthly.post_covid[data_start:last_date],
                             name='Actual Unemployment',
                            marker=dict(color='green', size=8)))

    # Plot post-covid projections
    fig.add_trace(go.Scatter(x=df_forecast_monthly[last_date:].index,
                             y=df_forecast_monthly.post_covid[last_date:],
                             mode='lines',
                             name='Post-Covid Projections'))

    fig.add_trace(go.Scatter(x=df_forecast_quarterly[last_date:].index.shift(periods=31, freq='D'),
                             y=df_forecast_quarterly.post_covid[last_date:],
                            mode='markers',
                            name='Post-Covid Quarterly Data',
                            marker=dict(color='red', size=8),
                            showlegend=False))

    # Plot pre-covid projections
    fig.add_trace(go.Scatter(x=df_forecast_quarterly[data_start:].index.shift(periods=31, freq='D'),
                             y=df_forecast_quarterly.pre_covid[data_start:],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='blue', size=8)))

    fig.add_trace(go.Scatter(x=df_forecast_monthly[data_start:].index,
                             y=df_forecast_monthly.pre_covid[data_start:],
                            name='Pre-Covid Projections',
                            marker=dict(color='blue')))

    # Add dashed line at last available date
    fig.add_vline(x=last_date, line_dash='dash')

    fig.update_layout(yaxis_title='Unemployment Rate')

    write_path = output_path + analysis_date + '/fig_2_unemp_full.pdf'
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
    fig_2(dfs, params, output_path)
