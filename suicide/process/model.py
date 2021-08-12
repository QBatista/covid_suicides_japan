"""
A script to generate figures for the model's predictions.

"""


import calendar
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# TODO(QBatista):
# 1. Add the age-gender analysis
# 2. Fix warnings
# 3. Unit testing

NOBS_MSG = 'Number of observations is different than expected number' + \
           ' of observations.'
COEF_MSG = 'Number of coefficients is different from the expected' + \
           ' number of coefficients'


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
    temp = ' observed versus expected number of suicides by month over time'
    title = suicide.name.capitalize() + temp
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
    pandemic_start = '2020-02-15'
    fig.add_vline(x=pandemic_start,
                  line_width=2,
                  line_dash="dash",
                  line_color="black",
                  name='Pandemic start')

    # Update title
    temp = ' observed versus expected number of suicides'
    title = suicide.name.capitalize() + temp
    fig.update_layout(title=title)

    return fig


def plot_induced_suicides_ts(pre_preds, suicide):
    covid_start = '2020-03'
    date_end = suicide.index[-1]

    data = suicide[covid_start:] - pre_preds[covid_start:date_end]

    fig = go.Figure()

    # Confidence intervals
    fig.add_trace(go.Bar(x=data.index,
                         y=data,
                         showlegend=False))

    temp = ' covid-induced suicides over time'
    title = suicide.name.capitalize() + temp
    fig.update_layout(title=title)

    return fig


def plot_explained_unemp_ts(pre_preds, post_preds, data_type):
    covid_start = '2020-03'

    data = post_preds[covid_start:] - pre_preds[covid_start:]

    fig = go.Figure()

    # Confidence intervals
    fig.add_trace(go.Bar(x=data.index,
                         y=data,
                         showlegend=False))

    temp = ' number of suicides explained by unemployment over time'
    title = data_type.capitalize() + temp
    fig.update_layout(title=title)

    return fig


def compute_key_numbers(pre_preds, post_preds, suicide):
    covid_start = '2020-03'
    present_date = suicide.index[-1]
    date_end = present_date + pd.Timedelta(366*3, unit='D')

    # Filter data
    pre_data = pre_preds[covid_start:date_end]
    post_data = post_preds[covid_start:date_end]
    suicide_data = suicide[covid_start:date_end]

    # Compute differences
    diff_actual = np.nansum(suicide_data - pre_data)
    diff_total = np.nansum(post_data - pre_data)
    diff_present = np.nansum(post_data[:present_date] - pre_data[:present_date])
    diff_future = diff_total - diff_present

    return diff_actual, diff_future, diff_present


def filter_dates(suicide, preds, date_start, date_end):
     # Unpack arguments
     pre_preds, post_preds, pre_conf_int = preds

     # Prepare args
     mask = pre_preds.index.isin(pre_preds[date_start:date_end].index)
     args = (pre_preds[date_start:date_end],
             post_preds[date_start:date_end],
             suicide.loc[date_start:date_end],
             pre_conf_int[mask, :])

     return args


def check_reg_res(res, date_start):
    # Test that the number of observations matches the expected
    # number of observations
    expected_nobs = ((2020 - int(date_start[:4])) * 12 + 2)
    if not res.nobs == expected_nobs:
        raise ValueError(NOBS_MSG)

    # Test that the number of coefficients matches the expected
    # number of coefficients
    expected_nb_coefs = 1 + 1 + 11 + 11 + 1
    if not len(res.params) == expected_nb_coefs:
        raise ValueError(COEF_MSG)


def gen_preds(suicide, unemp, forecasts, dates, α):
    # Unpack arguments
    date_start, date_end = dates

    # Construct training data
    X_pre_covid, X_post_covid = construct_Xs(unemp, forecasts)

    # Filter data
    y_train = suicide[date_start:date_end]
    X_train = X_pre_covid[date_start:date_end]

    # Run regression
    model = sm.regression.linear_model.OLS(y_train, X_train)
    res = model.fit()
    check_reg_res(res, date_start)

    # Generate predictions
    pre_preds = res.predict(X_pre_covid).rename('pred')
    post_preds = res.predict(X_post_covid).rename('pred')

    pre_conf_int = res.get_prediction(X_pre_covid).conf_int(alpha=α)

    return pre_preds, post_preds, pre_conf_int


def gen_figs(suicide, preds, path, analysis_date, date_start, data_type,
             group):

    date_end = suicide.index[-1]
    args = filter_dates(suicide, preds, date_start, date_end)
    fig = plot_preds_by_month(*args[:-1])
    full_path = path + '/present/unemp_by_month.pdf'
    fig.write_image(full_path, format='pdf')

    plot_start = '2019-01'
    args = filter_dates(suicide, preds, plot_start, date_end)
    fig = plot_preds_ts(*args)
    full_path = path + '/present/unemp_ts.pdf'
    fig.write_image(full_path, format='pdf')

    fig = plot_induced_suicides_ts(args[0], args[2])
    full_path = path + '/present/induced_suicides_ts.pdf'
    fig.write_image(full_path, format='pdf')

    fig = plot_explained_unemp_ts(args[0], args[1], data_type)
    full_path = path + '/present/explained_unemp_ts.pdf'
    fig.write_image(full_path, format='pdf')

    date_end += pd.Timedelta(366*3, unit='D')
    args = filter_dates(suicide, preds, date_start, date_end)
    fig = plot_preds_by_month(*args[:-1])
    full_path = path + '/full/unemp_by_month.pdf'
    fig.write_image(full_path, format='pdf')

    args = filter_dates(suicide, preds, plot_start, date_end)
    fig = plot_preds_ts(*args)
    full_path = path + '/full/unemp_ts.pdf'
    fig.write_image(full_path, format='pdf')

    fig = plot_explained_unemp_ts(args[0], args[1], data_type)
    full_path = path + '/full/explained_unemp_ts.pdf'
    fig.write_image(full_path, format='pdf')


def save_output(suicide, preds, path, analysis_date, date_start, data_type,
                group):

    # Unpack arguments
    pre_preds, post_preds, pre_conf_int = preds

    pre_preds.to_csv(path + '/pre_preds.csv')
    post_preds.to_csv(path + '/post_preds.csv')

    key_nb = compute_key_numbers(pre_preds, post_preds, suicide)

    key_nb_types = ('actual_minus_pre', 'post_minus_pre_future',
                    'post_minus_pre_present')

    df_key_nb = pd.DataFrame(key_nb, index=key_nb_types)
    df_key_nb.round().to_csv(path + '/key_nb.csv', header=False)


def initialize_dfs_store(data_types, key_nb_types, dates_start):
    cols = pd.MultiIndex.from_tuples(tuple((data_type, key_nb_type)
                                           for data_type in data_types
                                           for key_nb_type in key_nb_types))
    df_key_numbers = pd.DataFrame(columns=cols,
                                  index=pd.to_datetime(dates_start))

    cols = pd.MultiIndex.from_tuples(tuple((date_start, data_type))
                                         for date_start in dates_start
                                           for data_type in data_types
                                           ).sort_values()


    df_pre_preds = pd.DataFrame(columns=cols)
    df_post_preds = pd.DataFrame(columns=cols)

    return df_key_numbers, df_pre_preds, df_post_preds


def construct_Xs(unemp, forecasts):
    # Construct data for regression
    same_data_end_date = pd.to_datetime('2020-02')
    diff_data_start = same_data_end_date + pd.Timedelta(1, unit='MS')

    # Use same data for pre and post covid paths (actual unemployment)
    # until same_data_end_date
    temp_df = pd.DataFrame(columns=['post_covid', 'pre_covid'],
                           index=unemp[:same_data_end_date].index)
    temp_df.post_covid = unemp[:same_data_end_date].values
    temp_df.pre_covid = unemp[:same_data_end_date].values

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


def run_model(dfs, params, output_path):
    # Unpack arguments
    (df_unemp_dist,
     df_suicide_dist,
     df_forecast_total,
     df_unemp_annual,
     df_suicide_annual,
     df_forecast_quarterly) = dfs

    analysis_date = params['analysis_date']
    factor = params['factor']

    # Parameters
    dates_start = ('2009-01',
                   '2010-01',
                   '2011-01',
                   '2012-01')
    data_types = ('total', 'male', 'female')
    groups = ('0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79',
              '80_99', 'total')

    α = 0.25
    train_date_end = '2020-02'



    for date_start in dates_start:
        dates = (date_start, train_date_end)

        # Aggregate forecasts
        for data_type in data_types:
            for group in groups:
                path = output_path + analysis_date + '/model/agg_forecast/' \
                 + data_type + '/' + group + '/' + date_start + '/'

                suicide = df_suicide_dist[data_type][group]
                unemp = df_unemp_dist.total.total
                forecasts = df_forecast_total

                preds = gen_preds(suicide, unemp, forecasts, dates, α)

                gen_figs(suicide, preds, path, analysis_date,
                         date_start, data_type, group)

                save_output(suicide, preds, path, analysis_date,
                            date_start, data_type, group)

        # Age-gender-specific forecasts
        for data_type in data_types:
            for group in groups:
                path = output_path + analysis_date + '/model/group_forecast/' \
                 + data_type + '/' + group + '/' + date_start + '/'

                suicide = df_suicide_dist[data_type][group]
                unemp = df_unemp_dist[data_type][group]

                X = sm.tsa.add_trend(df_unemp_dist.total.total[:'2020-02'],
                                     trend='ct')
                y = unemp[:'2020-02']
                model = sm.regression.linear_model.OLS(y, X)
                res = model.fit()
                forecasts = df_forecast_total.apply(lambda x: res.predict(sm.tsa.add_trend(x, trend='ct')))

                preds = gen_preds(suicide, unemp, forecasts, dates, α)

                gen_figs(suicide, preds, path, analysis_date,
                         date_start, data_type, group)

                save_output(suicide, preds, path, analysis_date,
                            date_start, data_type, group)



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
    run_model(dfs, params, output_path)
