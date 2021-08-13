"""
A script to generate the full actual unemployment vs. forecasts plot.

"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .model import plot_unemp_suicide


def visualize_data(dfs, params, output_path):
    # Unpack arguments
    (df_unemp_dist,
     df_suicide_dist,
     df_forecast_total,
     df_unemp_annual,
     df_suicide_annual,
     df_forecast_quarterly) = dfs
    analysis_date = params['analysis_date']
    factor = params['factor']

    start_date = '1991-01'
    path = os.path.join(output_path, analysis_date, 'data_visualization', 'annual_total_ts_scatter.pdf')
    fig = plot_unemp_suicide(df_suicide_annual.loc[start_date:, 'total'],
                             df_unemp_annual.loc[start_date:, 'total'])
    fig.write_image(path, format='pdf')

    path = os.path.join(output_path, analysis_date, 'data_visualization', 'annual_male_ts_scatter.pdf')
    fig = plot_unemp_suicide(df_suicide_annual.loc[start_date:, 'male'],
                             df_unemp_annual.loc[start_date:, 'male'])
    fig.write_image(path, format='pdf')

    path = os.path.join(output_path, analysis_date, 'data_visualization', 'annual_female_ts_scatter.pdf')
    fig = plot_unemp_suicide(df_suicide_annual.loc[start_date:, 'female'],
                             df_unemp_annual.loc[start_date:, 'female'])
    fig.write_image(path, format='pdf')


if __name__ == '__main__':
    import yaml
    from load_data import load_data

    params_path = os.path.join(os.pardir, 'parameters.yml')
    output_path = os.path.join(os.pardir, 'output')
    clean_data_path = os.path.join(os.pardir, 'clean_data')

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    dfs = load_data(params, clean_data_path)
    visualize_data(dfs, params, output_path)
