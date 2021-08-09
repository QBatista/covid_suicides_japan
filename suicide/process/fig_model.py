"""
A script to generate figures for the model's predictions.

"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


NOBS_MSG = 'Number of observations is different than expected number' + \
           ' of observations.'
COEF_MSG = 'Number of coefficients is different from the expected' + \
           ' number of coefficients'


def gen_preds_by_month_plot(pre_covid_preds, post_covid_preds, df_suicide,
                            date_start, date_end, data_type):
    # Filter dates
    suicide_data = df_suicide[date_start:date_end]
    pre_data = pre_covid_preds[date_start:date_end]
    post_data = post_covid_preds[date_start:date_end]

    # Manual groupby for better plotting
    month_inds = sorted(suicide_data.index.month.unique())
    fig = make_subplots(4, 3, subplot_titles=month_inds)

    # Add traces for each month
    for month_idx in month_inds:
        # Hash to determine row and col indices
        row_idx = (month_idx - 1) // 3 + 1
        col_idx = (month_idx - 1) % 3 + 1
        showlegend = True if (row_idx == 1) & (col_idx == 1) else False

        suicide_month = suicide_data[suicide_data.index.month == month_idx]
        pre_month = pre_data[pre_data.index.month == month_idx]
        post_month = post_data[post_data.index.month == month_idx]

        # Actual suicides
        fig.add_trace(go.Scatter(x=suicide_month.index,
                                 y=suicide_month[data_type],
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
    title = data_type.capitalize() + temp
    fig.update_layout(height=1200,
                      width=900,
                      title=title)

    return fig


def gen_preds_ts_plot(pre_covid_preds, post_covid_preds, df_suicide, date_end,
                      data_type, pre_conf_int, post_conf_int):
    date_start = '2019-01'

    # Filter data
    suicide_data = df_suicide[date_start:date_end]
    pre_data = pre_covid_preds[date_start:date_end]
    post_data = post_covid_preds[date_start:date_end]

    mask = pre_covid_preds.index.isin(pre_data.index)

    fig = go.Figure()

    # Confidence intervals
    fig.add_trace(go.Scatter(x=pre_data.index,
                             y=pre_conf_int[mask, 0],  # Lower
                             fillcolor='rgba(0,0,255,0.1)',
                             line=dict(color='rgba(0,0,255,0.)'),
                             hoverinfo="skip",
                             showlegend=False))

    fig.add_trace(go.Scatter(x=pre_data.index,
                             y=pre_conf_int[mask, 1],  # Upper
                             fill='tonexty',
                             fillcolor='rgba(0,0,255,0.1)',
                             line=dict(color='rgba(0,0,255,0.)'),
                             hoverinfo="skip",
                             showlegend=False))

    # Lines
    fig.add_trace(go.Scatter(x=suicide_data.index,
                             y=suicide_data[data_type],
                             name='Observed',
                             mode='markers',
                             marker=dict(color='red')))

    fig.add_trace(go.Scatter(x=data_pre.index,
                             y=data_pre,
                             name='Expected (Pre-Covid)',
                             marker=dict(color='blue')))

    fig.add_trace(go.Scatter(x=pre_data.index,
                             y=post_data,
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
    title = data_type.capitalize() + temp
    fig.update_layout(title=title)

    return fig


def compute_key_numbers(pre_covid_preds, post_covid_preds, df_suicide,
                        data_type):
    covid_start = '2020-03'
    date_end = df_suicide.index[-1]

    # Filter data
    suicide_data = df_suicide[covid_start:][data_type]
    pre_data = pre_covid_preds[covid_start:]
    post_data = post_data[covid_start:]

    # Compute differences
    actual_diff = np.nansum(suicide_data - pre_data)
    total_diff = np.nansum(post_data - pre_data)
    diff_up_to_present = np.nansum(post_data[:date_end] - pre_data[:date_end])
    future_diff = total_diff - diff_up_to_present

    return actual_diff, future_diff, diff_up_to_present


def check_reg_res(res):
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


def fig_model(dfs, params, output_path):
    # Unpack arguments
    (df_unemp_monthly,
     df_unemp_annual,
     df_forecast_quarterly,
     df_forecast_monthly,
     df_suicide_monthly,
     df_suicide_annual) = dfs
    analysis_date = params['analysis_date']
    factor = params['factor']

    # Parameters
    α = 0.25

    dates_start = ('2008-01',
                   '2009-01',
                   '2010-01',
                   '2011-01',
                   '2012-01')

    data_types = ('total', 'male', 'female')
    key_nb_types = ('actual_minus_pre_covid', 'post_minus_pre_future',
                    'post_minus_pre_present', )

    train_date_end = '2020-02'

    # Construct data for regression
    same_data_end_date = pd.to_timestamp('2020-02')
    diff_data_start = same_data_end_date + pd.Timedelta(1, unit='MS')

    # Use same data for pre and post covid paths (actual unemployment)
    # until same_data_end_date
    temp_df = pd.DataFrame(columns=['post_covid', 'pre_covid'],
                           index=df_unemp_monthly[:same_data_end_date].index)
    temp_df.post_covid = df_unemp_monthly[:same_data_end_date].values
    temp_df.pre_covid = df_unemp_monthly[:same_data_end_date].values

    # Combine with "pre-covid" and "post-covid"' paths data
    df_unemp_wforecast = pd.concat([temp_df,
                                    df_forecast_monthly[diff_data_start:]])

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

    cols = pd.MultiIndex.from_tuples(tuple((data_type, key_nb_type)
                                           for data_type in data_types
                                           for key_nb_type in key_nb_types))
    df_key_numbers = pd.DataFrame(columns=cols,
                                  index=pd.to_datetime(dates_start))

    for date_start in dates_start:
        for data_type in data_types:
            # Filter data
            y_train = df_suicide_monthly[data_type][date_start:train_date_end]
            X_train = X_pre_covid[date_start:train_date_end]

            # Run regression
            model = sm.regression.linear_model.OLS(y_train, X_train)
            res = model.fit()
            check_reg_res(res)

            # Generate predictions
            pre_covid_preds = res.predict(X_pre_covid).rename('pred')
            post_covid_preds = res.predict(X_post_covid).rename('pred')

            pre_conf_int = res.get_prediction(X_pre_covid).conf_int(alpha=α)
            post_conf_int = res.get_prediction(X_post_covid).conf_int(alpha=α)

            # Generate figures
            p_start = output_path + analysis_date
            p_end = data_type + '_' + date_start + '.pdf'

            date_end = '2021-06'
            fig = gen_preds_by_month_plot(pre_covid_preds, post_covid_preds,
                                          df_suicide_monthly, date_start,
                                          date_end, data_type)
            path = p_start + '/unemp/present/unemp_by_month_' + p_end
            fig.write_image(path, format='pdf')

            fig = gen_preds_ts_plot(pre_covid_preds, post_covid_preds,
                                    df_suicide_monthly, date_end, data_type,
                                    pre_conf_int, post_conf_int)
            path = p_start + '/unemp/present/unemp_ts_' + p_end
            fig.write_image(path, format='pdf')

            date_end = '2024-12'
            fig = gen_preds_by_month_plot(pre_covid_preds, post_covid_preds,
                                          df_suicide_monthly, date_start,
                                          date_end, data_type)
            path = p_start + '/unemp/future/unemp_by_month_' + p_end
            fig.write_image(path, format='pdf')

            fig = gen_preds_ts_plot(pre_covid_preds, post_covid_preds,
                                    df_suicide_monthly, date_end, data_type,
                                    pre_conf_int, post_conf_int)
            path = p_start + '/unemp/future/unemp_ts_' + p_end
            fig.write_image(path, format='pdf')

            # Record key numbers
            df_key_numbers.loc[date_start, (data_type, )] = \
                compute_key_numbers(pre_covid_preds, post_covid_preds,
                                    df_suicide_monthly, data_type)

    df_key_numbers.to_csv(output_path + analysis_date + '/key_numbers.csv')


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
    fig_model(dfs, params, output_path)
