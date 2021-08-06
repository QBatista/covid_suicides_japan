"""
A script to generate the full actual unemployment vs. forecasts plot.

"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def unemp_suicide_plot(df_suicide, df_unemp, data_type='total'):
    color = df_suicide.index.year + (df_suicide.index.month - 1) / 12

    fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {}]])

    x = df_unemp.index.intersection(df_suicide.index)
    data_type_cap = data_type.capitalize()

    fig.add_trace(go.Scatter(x=x,
                                y=df_unemp[data_type],
                            name=data_type_cap + ' Unemployment Rate (Left)',
                            marker=dict(color='blue')),
                     row=1,
                     col=1)

    fig.add_trace(go.Scatter(x=x,
                                y=df_suicide[data_type],
                                name=data_type_cap + ' Number of Suicides (Right)',
                                marker=dict(color='red')),
                     secondary_y=True,
                     row=1,
                     col=1)

    fig.add_trace(go.Scatter(x=df_unemp[data_type],
                                y=df_suicide[data_type],
                                mode='markers',
                                marker=dict(color=color, colorbar=dict(thickness=10)),
                                showlegend=False), row=1, col=2)
    fig.update_layout(height=600,
                      width=1200)

    # Update xaxis properties
    fig.update_xaxes(title_text=data_type_cap + " Unemployment Rate", row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text=data_type_cap + " Number of Suicides", row=1, col=2)

    fig.update_layout(legend=dict(
        yanchor="top",
        y=1.15,
        xanchor="left",
        x=0.01
    ))

    return fig


def fig_3_to_8(dfs, params, output_path):
    # Unpack arguments
    (df_unemp_monthly,
     df_unemp_annual,
     df_forecast_quarterly,
     df_forecast_monthly,
     df_suicide_monthly,
     df_suicide_annual) = dfs
    analysis_date = params['analysis_date']
    factor = params['factor']

    # Figure 3
    start_date = '2008-01'
    path = output_path + analysis_date + '/fig_3_ts_scatter_total.pdf'
    fig = unemp_suicide_plot(df_suicide_monthly[start_date:], df_unemp_monthly[start_date:], data_type='total')
    fig.write_image(path, format='pdf')

    # Figure 4
    path = output_path + analysis_date + '/fig_4_ts_scatter_male.pdf'
    fig = unemp_suicide_plot(df_suicide_monthly[start_date:], df_unemp_monthly[start_date:], data_type='male')
    fig.write_image(path, format='pdf')

    # Figure 5
    path = output_path + analysis_date + '/fig_5_ts_scatter_female.pdf'
    fig = unemp_suicide_plot(df_suicide_monthly[start_date:], df_unemp_monthly[start_date:], data_type='female')
    fig.write_image(path, format='pdf')

    # Figure 6
    start_date = '1991-01'
    path = output_path + analysis_date + '/fig_6_ts_scatter_total_annual.pdf'
    fig = unemp_suicide_plot(df_suicide_annual[start_date:], df_unemp_annual[start_date:], data_type='total')
    fig.write_image(path, format='pdf')

    # Figure 7
    path = output_path + analysis_date + '/fig_7_ts_scatter_male_annual.pdf'
    fig = unemp_suicide_plot(df_suicide_annual[start_date:], df_unemp_annual[start_date:], data_type='male')
    fig.write_image(path, format='pdf')

    # Figure 8
    path = output_path + analysis_date + '/fig_8_ts_scatter_female_annual.pdf'
    fig = unemp_suicide_plot(df_suicide_annual[start_date:], df_unemp_annual[start_date:], data_type='female')
    fig.write_image(path, format='pdf')


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
    fig_3_to_8(dfs, params, output_path)
