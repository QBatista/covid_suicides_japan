"""
A script to generate figures for the model's predictions.

"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def gen_preds_by_month_plot(pre_covid_preds, post_covid_preds, df_suicide, date_start, date_end, data_type):
    # Filter dates
    sub_df_suicide = df_suicide[date_start:date_end]
    sub_df_pre_preds = pre_covid_preds[date_start:date_end]
    sub_df_post_preds = post_covid_preds[date_start:date_end]

    # Manual groupby for better plotting
    month_inds = sorted(sub_df_suicide.index.month.unique())
    fig = make_subplots(4, 3, subplot_titles=sub_df_suicide.index.month_name().unique())

    # Add traces for each month
    for month_idx in month_inds:
        # Hash
        row_idx = (month_idx - 1) // 3 + 1
        col_idx =  (month_idx - 1) % 3 + 1
        showlegend = True if (row_idx==1) & (col_idx==1) else False

        data_actual = sub_df_suicide[sub_df_suicide.index.month == month_idx]
        data_pre = sub_df_pre_preds[sub_df_pre_preds.index.month == month_idx]
        data_post = sub_df_post_preds[sub_df_post_preds.index.month == month_idx]

        # Actual suicides
        fig.add_trace(go.Scatter(x=data_actual.index,
                                 y=data_actual[data_type],
                                 name='Observed',
                                 mode='markers',
                                 marker=dict(color='red'),
                                 showlegend=showlegend),

                    row=row_idx,
                    col=col_idx)

        fig.add_trace(go.Scatter(x=data_pre.index,
                                 y=data_pre,
                                 name='Expected (Pre-Covid)',
                                 marker=dict(color='blue'),
                                 showlegend=showlegend),
                    row=row_idx,
                    col=col_idx)

        fig.add_trace(go.Scatter(x=data_post.index,
                                 y=data_post,
                                 name='Expected (Post-Covid)',
                                 marker=dict(color='green'),
                                 showlegend=showlegend),
                    row=row_idx,
                    col=col_idx)

    for i in range(1,13):
        fig['layout']['xaxis{}'.format(i)]['title'] = 'Year'

    fig.update_layout(height=1200,
                      width=900,
                      title= data_type.capitalize() + ' observed versus expected number of suicides by month over time')

    return fig


def gen_preds_ts_plot(pre_covid_preds, post_covid_preds, df_suicide, date_end, data_type, pre_conf_int, post_conf_int):
    date_start = '2019-01'

    data_actual = df_suicide[date_start:date_end][data_type]
    data_pre = pre_covid_preds[date_start:date_end]
    data_post = post_covid_preds[date_start:date_end]

    mask = pre_covid_preds.index.isin(pre_covid_preds[date_start:date_end].index)

    fig = go.Figure()

    # Confidence intervals
    fig.add_trace(go.Scatter(x=data_pre.index,
                             y=pre_conf_int[mask, 0],
                             fillcolor='rgba(0,0,255,0.1)',
                             line=dict(color='rgba(0,0,255,0.)'),
                             hoverinfo="skip",
                             showlegend=False))

    fig.add_trace(go.Scatter(x=data_pre.index,
                             y=pre_conf_int[mask, 1],
                             fill='tonexty',
                             fillcolor='rgba(0,0,255,0.1)',
                             line=dict(color='rgba(0,0,255,0.)'),
                             hoverinfo="skip",
                             showlegend=False))

    # Lines
    fig.add_trace(go.Scatter(x=data_actual.index,
                             y=data_actual,
                             name='Observed',
                             mode='markers',
                             marker=dict(color='red')))

    fig.add_trace(go.Scatter(x=data_pre.index,
                             y=data_pre,
                             name='Expected (Pre-Covid)',
                             marker=dict(color='blue')))

    fig.add_trace(go.Scatter(x=data_pre.index,
                             y=data_post,
                             name='Expected (Post-Covid)',
                             marker=dict(color='green')))

    pandemic_start = '2020-02-15'
    fig.add_vline(x=pandemic_start,
                  line_width=2,
                  line_dash="dash",
                  line_color="black",
                  name='Pandemic start')

    fig.update_layout(title= data_type.capitalize() + ' observed versus expected number of suicides')

    return fig


def compute_key_numbers(pre_covid_preds, post_covid_preds, df_suicide, data_type):
    covid_start = '2020-03'
    date_end = df_suicide.index[-1]
    actual_diff = np.nansum(df_suicide[covid_start:][data_type] - pre_covid_preds[covid_start:])
    total_diff = np.nansum(post_covid_preds[covid_start:] - pre_covid_preds[covid_start:])
    diff_up_to_last = np.nansum(post_covid_preds[covid_start:date_end] - pre_covid_preds[covid_start:date_end])
    future_diff = total_diff - diff_up_to_last

    return actual_diff, future_diff, diff_up_to_last


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


    max_lags = 0

    # Construct data for regression
    same_data_end_date = '2020-02'

    # Use same data (actual unemployment) until same_data_end_date
    temp_df = pd.DataFrame(columns=['post_covid', 'pre_covid'], index=df_unemp_monthly[:same_data_end_date].index)
    temp_df.post_covid = df_unemp_monthly[:same_data_end_date].values
    temp_df.pre_covid = df_unemp_monthly[:same_data_end_date].values

    # Combine with "pre-covid" and "post-covid"' paths data
    df_unemp_wforecast = pd.concat([temp_df, df_forecast_monthly['2020-03':]])

    # Get year indices
    years = df_unemp_wforecast.index.year

    # Get month dummies
    month_dummies = pd.get_dummies(df_unemp_wforecast.index.month)
    interactions = month_dummies.iloc[:, 1:] * years.values.reshape((-1,1))

    # Get lagged variables
    df_unemp_wlags = pd.concat([df_unemp_wforecast[path_type].shift(lag).rename((path_type, 'unemp_' + str(lag)))
                                for path_type in df_unemp_wforecast.columns
                                 for lag in range(max_lags+1)], axis=1)

    # Combine data
    X1 = pd.concat([years.to_series(index=month_dummies.index), month_dummies, interactions], axis=1)
    X1.index = df_unemp_wforecast.index

    X2_pre_covid = df_unemp_wlags['pre_covid']
    X2_post_covid = df_unemp_wlags['post_covid']

    X_pre_covid = X1.join(X2_pre_covid)
    X_post_covid = X1.join(X2_post_covid)

    alpha = 0.25

    dates_start = ('2008-01',
                  '2009-01',
                  '2010-01',
                  '2011-01',
                  '2012-01')

    data_types = ('total', 'male', 'female')
    key_nb_types = ('actual_minus_pre_covid', 'post_minus_pre_future', 'post_minus_pre_present', )

    cols = pd.MultiIndex.from_tuples(tuple((data_type, key_nb_type)
                                           for data_type in data_types
                                           for key_nb_type in key_nb_types)).sort_values()
    df_key_numbers = pd.DataFrame(columns=cols,
                                  index=pd.to_datetime(dates_start))

    for date_start in dates_start:
        for data_type in data_types:
            train_date_end = '2020-02'
            y_train = df_suicide_monthly[data_type][date_start:train_date_end]
            X_train = X_pre_covid[date_start:train_date_end]

            model = sm.regression.linear_model.OLS(y_train, X_train)
            res = model.fit()

            pre_covid_preds = res.predict(X_pre_covid).rename('pred')
            post_covid_preds = res.predict(X_post_covid).rename('pred')

            pre_conf_int = res.get_prediction(X_pre_covid).conf_int(alpha=alpha)
            post_conf_int = res.get_prediction(X_post_covid).conf_int(alpha=alpha)

            # Test that the number of observations matches the expected number of observations
            expected_nobs = ((2020 - int(date_start[:4])) * 12 + 2)
            if not res.nobs == expected_nobs:
                raise ValueError('Number of observations is different than expected number of observations.')

            # Test that the number of coefficients matches the expected number of coefficients
            expected_nb_coefs = 1 + 1 + 11 + 11 + 1
            if not len(res.params) == expected_nb_coefs:
                raise ValueError('Number of coefficients is different from the expected number of coefficients')

            date_end = '2021-06'
            fig = gen_preds_by_month_plot(pre_covid_preds, post_covid_preds, df_suicide_monthly, date_start, date_end, data_type)
            path = output_path + analysis_date + '/unemp/present/unemp_by_month_'  + data_type + '_' + date_start + '.pdf'
            fig.write_image(path, format='pdf')

            fig = gen_preds_ts_plot(pre_covid_preds, post_covid_preds, df_suicide_monthly, date_end, data_type, pre_conf_int, post_conf_int)
            path = output_path + analysis_date + '/unemp/present/unemp_ts_'  + data_type + '_' + date_start + '.pdf'
            fig.write_image(path, format='pdf')

            date_end = '2024-12'
            fig = gen_preds_by_month_plot(pre_covid_preds, post_covid_preds, df_suicide_monthly, date_start, date_end, data_type)
            path = output_path + analysis_date + '/unemp/future/unemp_by_month_'  + data_type + '_' + date_start + '.pdf'
            fig.write_image(path, format='pdf')

            fig = gen_preds_ts_plot(pre_covid_preds, post_covid_preds, df_suicide_monthly, date_end, data_type, pre_conf_int, post_conf_int)
            path = output_path + analysis_date + '/unemp/future/unemp_ts_'  + data_type + '_' + date_start + '.pdf'
            fig.write_image(path, format='pdf')

            df_key_numbers.loc[date_start, (data_type, )] = compute_key_numbers(pre_covid_preds, post_covid_preds, df_suicide_monthly, data_type)

    df_key_numbers.to_html(output_path + analysis_date + '/key_numbers.html')


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
