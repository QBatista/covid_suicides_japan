"""
A script to load the data for processing.

"""

import os
import pandas as pd
import numpy as np


def load_unemp_data(params, clean_data_path):
    # Unpack parameters
    analysis_date = params['analysis_date']
    factor = params['factor']

    # Load distribution data
    dist_path = os.path.join(clean_data_path, analysis_date, 'unemp_dist.csv')
    df_unemp_dist = pd.read_csv(dist_path, index_col=0, header=[0, 1])
    idx = pd.to_datetime(df_unemp_dist.index).to_period('M').to_timestamp()
    df_unemp_dist.index = idx

    # Load annual unemployment data
    annual_path = os.path.join(clean_data_path, analysis_date, 'unemp_annual.csv')
    df_unemp_annual = pd.read_csv(annual_path, index_col=0)
    df_unemp_annual.index = pd.to_datetime(df_unemp_annual.index)

    return df_unemp_dist, df_unemp_annual


def load_suicide_data(params, clean_data_path):
    # Unpack parameters
    analysis_date = params['analysis_date']
    factor = params['factor']

    # Load distribution suicide data
    dist_path = os.path.join(clean_data_path, analysis_date, 'suicide_dist.csv')
    df_suicide_dist = pd.read_csv(dist_path, index_col=0, header=[0, 1])
    df_suicide_dist.index = pd.to_datetime(df_suicide_dist.index)

    # Load annual suicide data
    annual_path = os.path.join(clean_data_path, analysis_date, 'suicide_annual.csv')
    df_suicide_annual = pd.read_csv(annual_path, index_col=0)
    df_suicide_annual.index = pd.to_datetime(df_suicide_annual.index)

    return df_suicide_dist, df_suicide_annual


def load_forecast_data(df_unemp_dist, params, clean_data_path):
    # Unpack parameters
    analysis_date = params['analysis_date']
    factor = params['factor']

    # Load forecast data
    forecast_path = os.path.join(clean_data_path, analysis_date, 'forecast.csv')
    df_forecast_quarterly = pd.read_csv(forecast_path, index_col=0)
    idx = pd.to_datetime(df_forecast_quarterly.index)
    idx = idx.to_period('Q').to_timestamp()
    df_forecast_quarterly.index = idx

    # Extend index to December 2024
    idx = pd.date_range(start=df_forecast_quarterly.index[0],
                        end='2024-12-31',
                        freq=df_forecast_quarterly.index.freq)
    df_forecast_quarterly = df_forecast_quarterly.reindex(idx)

    # Resample to monthly frequency
    df_forecast_total = df_forecast_quarterly.to_period('Q').resample('1M')
    df_forecast_total = df_forecast_total.asfreq()

    # Shift so that interpolation node is middle month
    df_forecast_total = df_forecast_total.shift()

    # Interpolate (and extrapolate at end points)
    df_forecast_total.interpolate(limit_direction='both', inplace=True)

    # Change index to timestamp instead of period
    df_forecast_total.index = df_forecast_total.index.to_timestamp()

    # Update the "post-covid" data by replacing interpolated values with actual
    # values
    first_date = df_forecast_total.index[0]
    last_date = df_unemp_dist.total.total.index[-1]
    unemp_data = df_unemp_dist.total.total[first_date:last_date].values.ravel()
    df_forecast_total.loc[first_date:last_date, 'post_covid'] = unemp_data

    # Decay post-covid extrapolation
    last_forecast = '2023-02'
    n = len(df_forecast_total.loc[last_forecast:, 'post_covid'])
    factors = [factor ** i for i in range(n)]
    df_forecast_total.loc[last_forecast:, 'post_covid'] *= factors

    return df_forecast_total, df_forecast_quarterly


def load_data(params, clean_data_path):
    (df_unemp_dist,
     df_unemp_annual) = load_unemp_data(params, clean_data_path)

    (df_suicide_dist,
     df_suicide_annual) = load_suicide_data(params, clean_data_path)

    (df_forecast_total,
     df_forecast_quarterly) = load_forecast_data(df_unemp_dist, params,
                                                 clean_data_path)

    dfs = (df_unemp_dist,
           df_suicide_dist,
           df_forecast_total,
           df_unemp_annual,
           df_suicide_annual,
           df_forecast_quarterly)

    return dfs
