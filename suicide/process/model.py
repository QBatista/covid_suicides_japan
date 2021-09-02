"""
A script to generate figures for the model's predictions.

"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from .util.plot import *


# TODO(QBatista):
# 1. Update `load_data`
# 2. Clean up the code
#
# - Unit testing
# - Documentation


NOBS_MSG = 'Number of observations is different than expected number' + \
           ' of observations.'
COEF_MSG = 'Number of coefficients is different from the expected' + \
           ' number of coefficients'
COVID_START = '2020-03'
LAST_TRAIN_DATE = '2020-02'  # Last date for fitting the model
ALPHA = 0.25  # Prediction confidence interval parameter
DATES_START = ('2009-01',
               '2010-01',
               '2011-01',
               '2012-01')
DATA_TYPES = ('total', 'male', 'female')
GROUPS = ('0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79',
          '80_99', 'total')
REG_NB_COEFS = 1 + 1 + 11 + 11 + 1


def compute_key_numbers(pre_preds, post_preds, suicide):
    """
    Compute the three key numbers for our analysis:
        1. the difference between observed and predicted suicides
        2. the number of additional covid-induced suicides we anticipate due
           to changes in unemployment
        3. the number of covid-induced suicides that can be explained due to
           changes in unemployment up to now.

    """

    present_date = suicide.index[-1]
    date_end = present_date + pd.Timedelta(366*3, unit='D')

    # Filter data
    pre_data = pre_preds[COVID_START:date_end]
    post_data = post_preds[COVID_START:date_end]
    suicide_data = suicide[COVID_START:date_end]

    # Compute differences
    diff_actual = np.nansum(suicide_data - pre_data)
    diff_total = np.nansum(post_data - pre_data)
    diff_present = np.nansum(post_data[:present_date] - pre_data[:present_date])
    diff_future = diff_total - diff_present

    return diff_actual, diff_future, diff_present


def filter_dates(suicide, preds, date_start, date_end):
    """
    Helper function to select `suicide` and `preds` data between
    `date_start` and `date_end`.

    """

    # Unpack arguments
    pre_preds, post_preds, pre_conf_int = preds

    # Prepare args
    mask = pre_preds.index.isin(pre_preds[date_start:date_end].index)
    out = (pre_preds[date_start:date_end],
           post_preds[date_start:date_end],
           suicide.loc[date_start:date_end],
           pre_conf_int[mask, :])

    return out


def check_reg_res(res, date_start):
    """
    Helper function for checking properties of the regression result object
    `res`.

    """

    # Test that the number of observations matches the expected
    # number of observations
    ds = pd.to_datetime(date_start)
    de = pd.to_datetime(LAST_TRAIN_DATE)
    expected_nobs = (de.year - ds.year) * 12 + de.month - ds.month + 1
    if not res.nobs == expected_nobs:
        raise ValueError(NOBS_MSG)

    # Test that the number of coefficients matches the expected
    # number of coefficients
    if not len(res.params) == REG_NB_COEFS:
        raise ValueError(COEF_MSG)


def gen_preds(suicide, unemp, forecasts, date_start):
    # Construct training data
    X_pre_covid, X_post_covid = construct_Xs(unemp, forecasts)

    # Filter data
    y_train = suicide[date_start:LAST_TRAIN_DATE]
    X_train = X_pre_covid[date_start:LAST_TRAIN_DATE]

    # Run regression
    model = sm.regression.linear_model.OLS(y_train, X_train)
    res = model.fit()
    check_reg_res(res, date_start)

    # Generate predictions
    pre_preds = res.predict(X_pre_covid).rename('pred')
    post_preds = res.predict(X_post_covid).rename('pred')

    pre_conf_int = res.get_prediction(X_pre_covid).conf_int(alpha=ALPHA)

    return (pre_preds, post_preds, pre_conf_int), res


def gen_figs(suicide, preds, forecasts, unemp, path, analysis_date, date_start,
             data_type, group, last_date):

    fig = plot_unemp_suicide(suicide, unemp)
    full_path = os.path.join(path, 'present', 'unemp_suicide_ts_scatter.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_unemployment(forecasts, last_date)
    full_path = os.path.join(path, 'present', 'unemp_vs_forecast.pdf')

    fig.write_image(full_path, format='pdf')

    fig = plot_forecasts(forecasts, last_date)
    full_path = os.path.join(path, 'full', 'forecasts.pdf')
    fig.write_image(full_path, format='pdf')

    date_end = suicide.index[-1]
    args = filter_dates(suicide, preds, date_start, date_end)
    fig = plot_preds_by_month(*args[:-1])
    full_path = os.path.join(path, 'present', 'unemp_by_month.pdf')
    fig.write_image(full_path, format='pdf')

    plot_start = '2019-01'
    args = filter_dates(suicide, preds, plot_start, date_end)
    fig = plot_preds_ts(*args)
    full_path = os.path.join(path, 'present', 'unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_induced_suicides_ts(args[0], args[2])
    full_path = os.path.join(path, 'present', 'induced_suicides_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_explained_unemp_ts(args[0], args[1], suicide.name)
    full_path = os.path.join(path, 'present', 'explained_unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    date_end += pd.Timedelta(366*3, unit='D')
    args = filter_dates(suicide, preds, date_start, date_end)
    fig = plot_preds_by_month(*args[:-1])
    full_path = os.path.join(path, 'full', 'unemp_by_month.pdf')
    fig.write_image(full_path, format='pdf')

    args = filter_dates(suicide, preds, plot_start, date_end)
    fig = plot_preds_ts(*args)
    full_path = os.path.join(path, 'full', 'unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_explained_unemp_ts(args[0], args[1], suicide.name)
    full_path = os.path.join(path, 'full', 'explained_unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')


def save_output(suicide, preds, path, analysis_date, date_start, data_type,
                group):
    # Unpack arguments
    pre_preds, post_preds, pre_conf_int = preds

    pre_preds.to_csv(os.path.join(path, 'pre_preds.csv'))
    post_preds.to_csv(os.path.join(path, 'post_preds.csv'))

    induced_suicide = compute_induced_suicides(suicide, pre_preds)
    induced_suicide.to_csv(os.path.join(path, 'induced_suicide.csv'))

    key_nb = compute_key_numbers(pre_preds, post_preds, suicide)

    key_nb_types = ('actual_minus_pre', 'post_minus_pre_future',
                    'post_minus_pre_present')

    df_key_nb = pd.DataFrame(key_nb, index=key_nb_types)
    df_key_nb.round().to_csv(os.path.join(path, 'key_nb.csv'), header=False)


def construct_Xs(unemp, forecasts):
    # Construct data for regression
    diff_data_start = pd.to_datetime(LAST_TRAIN_DATE) + pd.Timedelta(1, unit='MS')

    # Use same data for pre and post covid paths (actual unemployment)
    # until LAST_TRAIN_DATE
    temp_df = pd.DataFrame(columns=['post_covid', 'pre_covid'],
                           index=unemp[:LAST_TRAIN_DATE].index)
    temp_df.post_covid = unemp[:LAST_TRAIN_DATE].values
    temp_df.pre_covid = unemp[:LAST_TRAIN_DATE].values

    # Combine with "pre-covid" and "post-covid"' paths data
    df_unemp_wforecast = pd.concat([temp_df, forecasts[diff_data_start:]])

    # Get year indices
    years = df_unemp_wforecast.index.year

    # Get month dummies
    month_dummies = pd.get_dummies(df_unemp_wforecast.index.month)
    interactions = month_dummies.iloc[:, 1:] * years.values.reshape((-1, 1))

    # Combine data
    X1 = pd.concat([years.to_series(index=month_dummies.index), month_dummies,
                    interactions], axis=1)
    X1.index = df_unemp_wforecast.index

    X2_pre_covid = df_unemp_wforecast['pre_covid']
    X2_post_covid = df_unemp_wforecast['post_covid']

    X_pre_covid = X1.join(X2_pre_covid)
    X_post_covid = X1.join(X2_post_covid)

    Xs = (X_pre_covid, X_post_covid)

    return Xs


def rename_series(suicide, unemp, data_type, group):
    name = data_type.capitalize()
    if group != 'total':
        name += ' (Age: ' + group.replace('_', '-') + ')'

    suicide.rename(name, inplace=True)
    unemp.rename(name, inplace=True)


def gen_group_forecasts(unemp, df_unemp_dist, df_forecast_total):
    X = sm.tsa.add_trend(df_unemp_dist.total.total[:LAST_TRAIN_DATE],
                         trend='ct')
    y = unemp[:LAST_TRAIN_DATE]
    model = sm.regression.linear_model.OLS(y, X)
    res = model.fit()
    def predict(x): return res.predict(sm.tsa.add_trend(x, trend='ct'))
    forecasts = df_forecast_total.apply(predict)

    return forecasts


def run_model(dfs, params, output_path):
    # Unpack arguments
    (df_unemp_dist,
     df_suicide_dist,
     df_forecast_total,
     df_unemp_annual,
     df_suicide_annual,
     df_forecast_quarterly) = dfs

    analysis_date = params['analysis_date']

    for date_start in DATES_START:
        # Aggregate forecasts
        for data_type in DATA_TYPES:
            for group in GROUPS:
                path = os.path.join(output_path, analysis_date, 'model',
                 'aggregate', data_type, group, date_start)

                suicide = df_suicide_dist[data_type][group]
                unemp = df_unemp_dist.total.total
                last_date = unemp.index[-1]
                forecasts = df_forecast_total

                rename_series(suicide, unemp, data_type, group)

                preds, res = gen_preds(suicide, unemp, forecasts, date_start)

                gen_figs(suicide, preds, forecasts, unemp, path, analysis_date,
                         date_start, data_type, group, last_date)

                save_output(suicide, preds, path, analysis_date,
                            date_start, data_type, group)
                res.save(os.path.join(path, 'regression_result.pickle'))

        # Age-gender-specific forecasts
        for data_type in DATA_TYPES:
            for group in GROUPS:
                path = os.path.join(output_path, analysis_date, 'model',
                 'group', data_type, group, date_start)

                suicide = df_suicide_dist[data_type][group]
                unemp = df_unemp_dist[data_type][group]
                last_date = unemp.index[-1]

                rename_series(suicide, unemp, data_type, group)

                forecasts = gen_group_forecasts(unemp, df_unemp_dist,
                                                df_forecast_total)

                preds, res = gen_preds(suicide, unemp, forecasts, date_start)

                gen_figs(suicide, preds, forecasts, unemp, path, analysis_date,
                         date_start, data_type, group, last_date)

                save_output(suicide, preds, path, analysis_date,
                            date_start, data_type, group)
                res.save(os.path.join(path, 'regression_result.pickle'))



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
    run_model(dfs, params, output_path)
