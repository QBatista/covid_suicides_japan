

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from load_data import load_suicide_data
from util.plot import *
from model import filter_dates


# TODO(QBatista):
# 1. Find better names for the folders
# 2. Use parameter for generating summaries


DATES_START = ('2009-01',
               '2010-01',
               '2011-01',
               '2012-01')
UNEMP_TYPES = ('aggregate', 'group')
AGE_GROUPS = ('0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79',
              '80_99')
DATA_TYPES = ('male', 'female', 'total')


def plot_distrib(data, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    fig.update_layout(title_text=title, width=1200, height=600)
    return fig


def plot_loss_life_exp(suicide, infections, title):
    suicide_total = suicide.sum() / 1000
    infections_total = infections.sum() / 1000

    name_s = 'Suicides (Total: approx. %iK)' % suicide_total
    name_i = 'Infections (Total: approx. %iK)' % infections_total

    fig = go.Figure()
    fig.add_trace(go.Bar(x=suicide.index, y=suicide, name=name_s))
    fig.add_trace(go.Bar(x=suicide.index, y=infections, name=name_i))

    fig.update_layout(title_text=title, width=1200, height=600)

    return fig


def load_data(params, date_start, unemp_type, output_path, clean_data_path):
    analysis_date = params['analysis_date']

    path = os.path.join(clean_data_path, analysis_date, 'life_expectancy.csv')
    df_life_exp = pd.read_csv(path, index_col=0)

    path = os.path.join(clean_data_path, analysis_date, 'infection_deaths.csv')
    df_infections = pd.read_csv(path, index_col=0)

    df_suicide_m = pd.DataFrame()
    df_suicide_f = pd.DataFrame()

    for group in AGE_GROUPS:
        path = os.path.join(output_path, analysis_date, 'model', unemp_type,
                            'male', group, date_start, 'induced_suicide.csv')
        df_suicide_m[group] = pd.read_csv(path, index_col=0).iloc[:, 0]

    for group in AGE_GROUPS:
        path = os.path.join(output_path, analysis_date, 'model', unemp_type,
                            'female', group, date_start, 'induced_suicide.csv')
        df_suicide_f[group] = pd.read_csv(path, index_col=0).iloc[:, 0]

    dfs = (df_life_exp, df_infections, df_suicide_m, df_suicide_f)

    return dfs


def plot_life_exp(params, date_start, unemp_type, output_path, clean_data_path):
    analysis_date = params['analysis_date']

    df_life_exp, df_infections, df_suicide_m, df_suicide_f = \
        load_data(params, date_start, unemp_type, output_path,
                  clean_data_path)

    suicide_m_le = df_suicide_m.multiply(df_life_exp['male'])
    suicide_m_sum = suicide_m_le.sum(axis=0).rename('Male')
    suicide_m_sum.index = [group.replace('_', ' to ') for group in suicide_m_sum.index]

    suicide_f_le = df_suicide_f.multiply(df_life_exp['female'])
    suicide_f_sum = suicide_f_le.sum(axis=0).rename('Female')
    suicide_f_sum.index = [group.replace('_', ' to ') for group in suicide_f_sum.index]

    suicide_mf_le = suicide_m_le + suicide_f_le
    suicide_le_total = (suicide_m_sum + suicide_f_sum).rename('Total')
    suicide_le_total.index = [group.replace('_', ' to ') for group in suicide_le_total.index]

    mask = df_infections.age_group.isin(AGE_GROUPS)
    infections_data = df_infections[mask].pivot(columns=['gender_group', 'age_group'], values='value')

    infections_m_le = infections_data['male'].multiply(df_life_exp['male'])
    infections_f_le = infections_data['female'].multiply(df_life_exp['female'])

    infections_m_le_sum = infections_m_le.sum(axis=0).rename('Male')
    infections_f_le_sum = infections_f_le.sum(axis=0).rename('Female')

    infections_mf_le = infections_m_le + infections_f_le

    infections_le_total = (infections_m_le_sum + infections_f_le_sum).rename('Total')

    # Lost life expectancy by age groups
    title_end = ' Years of Life Expectancy Lost Due to Covid Induced Suicides vs. Infections by Age Group'
    title = suicide_le_total.name + title_end
    fig = plot_loss_life_exp(suicide_le_total, infections_le_total, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'total', date_start, 'lost_life_exp_by_age.pdf')
    fig.write_image(full_path, format='pdf')

    title = suicide_m_sum.name + title_end
    fig = plot_loss_life_exp(suicide_m_sum, infections_m_le_sum, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'male', date_start, 'lost_life_exp_by_age.pdf')
    fig.write_image(full_path, format='pdf')

    title = suicide_f_sum.name + title_end
    fig = plot_loss_life_exp(suicide_f_sum, infections_f_le_sum, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'female', date_start, 'lost_life_exp_by_age.pdf')
    fig.write_image(full_path, format='pdf')

    # Lost life expectancy over time
    title = 'Total Years of Life Expectancy Lost Due to Covid Induced Suicides vs. Infections Over Time'
    fig = plot_loss_life_exp(suicide_mf_le.sum(axis=1), infections_mf_le.sum(axis=1), title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'total', date_start, 'lost_life_exp_over_time.pdf')
    fig.write_image(full_path, format='pdf')

    title = 'Male Years of Life Expectancy Lost Due to Covid Induced Suicides vs. Infections Over Time'
    fig = plot_loss_life_exp(suicide_m_le.sum(axis=1), infections_m_le.sum(axis=1), title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'male', date_start, 'lost_life_exp_over_time.pdf')
    fig.write_image(full_path, format='pdf')

    title = 'Female Years of Life Expectancy Lost Due to Covid Induced Suicides vs. Infections Over Time'
    fig = plot_loss_life_exp(suicide_f_le.sum(axis=1), infections_f_le.sum(axis=1), title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'female', date_start, 'lost_life_exp_over_time.pdf')
    fig.write_image(full_path, format='pdf')


def plot_deaths(params, date_start, unemp_type, output_path, clean_data_path):
    analysis_date = params['analysis_date']
    df_life_exp, df_infections, df_suicide_m, df_suicide_f = \
        load_data(params, date_start, unemp_type, output_path,
                  clean_data_path)

    suicide_m_sum = df_suicide_m.sum(axis=0).rename('Male')
    suicide_m_sum.index = [group.replace('_', ' to ') for group in suicide_m_sum.index]

    suicide_f_sum = df_suicide_f.sum(axis=0).rename('Female')
    suicide_f_sum.index = [group.replace('_', ' to ') for group in suicide_f_sum.index]

    suicide_mf = df_suicide_m + df_suicide_f
    suicide_total = (suicide_m_sum + suicide_f_sum).rename('Total')
    suicide_total.index = [group.replace('_', ' to ') for group in suicide_total.index]

    mask = df_infections.age_group.isin(AGE_GROUPS)
    infections_data = df_infections[mask].pivot(columns=['gender_group', 'age_group'], values='value')

    infections_m = infections_data['male']
    infections_f = infections_data['female']

    infections_m_sum = infections_m.sum(axis=0).rename('Male')
    infections_f_sum = infections_f.sum(axis=0).rename('Female')

    infections_mf = infections_m + infections_f

    infections_total = (infections_m_sum + infections_f_sum).rename('Total')

    title_end = ' Covid Induced Suicides vs. Infection Deaths by Age Group'
    title = suicide_total.name + title_end
    fig = plot_loss_life_exp(suicide_total, infections_total, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'total', date_start, 'suicides_vs_infections.pdf')
    fig.write_image(full_path, format='pdf')

    title_end = ' Suicides Distribution'
    title = suicide_total.name + title_end
    fig = plot_distrib(suicide_total, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'total', date_start, 'suicides_dist.pdf')
    fig.write_image(full_path, format='pdf')

    title_end = ' Infections Distribution'
    title = suicide_total.name + title_end
    fig = plot_distrib(infections_total, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'total', date_start, 'infections_dist.pdf')
    fig.write_image(full_path, format='pdf')

    title = suicide_m_sum.name + title_end
    fig = plot_loss_life_exp(suicide_m_sum, infections_m_sum, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'male', date_start, 'suicides_vs_infections.pdf')
    fig.write_image(full_path, format='pdf')

    title_end = ' Suicides Distribution'
    title = suicide_total.name + title_end
    fig = plot_distrib(suicide_m_sum, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'male', date_start, 'suicides_dist.pdf')
    fig.write_image(full_path, format='pdf')

    title_end = ' Infections Distribution'
    title = suicide_total.name + title_end
    fig = plot_distrib(infections_m_sum, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'male', date_start, 'infections_dist.pdf')
    fig.write_image(full_path, format='pdf')

    title = suicide_f_sum.name + title_end
    fig = plot_loss_life_exp(suicide_f_sum, infections_f_sum, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'female', date_start, 'suicides_vs_infections.pdf')
    fig.write_image(full_path, format='pdf')

    title_end = ' Suicides Distribution'
    title = suicide_total.name + title_end
    fig = plot_distrib(suicide_f_sum, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'female', date_start, 'suicides_dist.pdf')
    fig.write_image(full_path, format='pdf')

    title_end = ' Infections Distribution'
    title = suicide_total.name + title_end
    fig = plot_distrib(infections_f_sum, title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'female', date_start, 'infections_dist.pdf')
    fig.write_image(full_path, format='pdf')

    title = 'Total Covid Induced Suicides vs. Infection Deaths Over Time'
    fig = plot_loss_life_exp(suicide_mf.sum(axis=1), infections_mf.sum(axis=1), title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'total', date_start, 'suicides_vs_infections_over_time.pdf')
    fig.write_image(full_path, format='pdf')

    title = 'Male Covid Induced Suicides vs. Infection Deaths Over Time'
    fig = plot_loss_life_exp(df_suicide_m.sum(axis=1), infections_m.sum(axis=1), title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'male', date_start, 'suicides_vs_infections_over_time.pdf')
    fig.write_image(full_path, format='pdf')

    title = 'Female Covid Induced Suicides vs. Infection Deaths Over Time'
    fig = plot_loss_life_exp(df_suicide_f.sum(axis=1), infections_f.sum(axis=1), title)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'female', date_start, 'suicides_vs_infections_over_time.pdf')
    fig.write_image(full_path, format='pdf')


def gen_figs(params, output_path, clean_data_path):
    analysis_date = params['analysis_date']

    for date_start in DATES_START:
        for unemp_type in UNEMP_TYPES:
            plot_life_exp(params, date_start, unemp_type, output_path, clean_data_path)
            plot_deaths(params, date_start, unemp_type, output_path, clean_data_path)


def summary_table(params, output_path):

    analysis_date = params['analysis_date']

    summary_df = pd.DataFrame(index=['aggregate', 'age_only', 'gender_only', 'age_gender'],
                              columns=['actual_minus_pre', 'post_minus_pre_future', 'post_minus_pre_present'])

    # Aggregate
    forecast_type = 'aggregate'
    data_type = 'total'
    age_group = 'total'
    start_date = '2010-01'

    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'key_nb.csv')
    summary_df.loc['aggregate'] = pd.read_csv(path, index_col=0, header=None).T.values


    # Age Only
    df = pd.DataFrame(index= AGE_GROUPS, columns=['actual_minus_pre', 'post_minus_pre_future', 'post_minus_pre_present'])

    forecast_type = 'group'
    data_type = 'total'

    for age_group in AGE_GROUPS:
        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'key_nb.csv')
        df.loc[age_group] = pd.read_csv(path, index_col=0, header=None).T.values

    summary_df.loc['age_only'] = df.sum(axis=0)

    # Gender only
    forecast_type = 'group'
    data_type = 'male'
    age_group = 'total'
    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'key_nb.csv')

    temp = pd.read_csv(path, index_col=0, header=None).T.values

    data_type = 'female'
    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'key_nb.csv')

    summary_df.loc['gender_only'] = temp + pd.read_csv(path, index_col=0, header=None).T.values

    # Age-gender
    df = pd.DataFrame(index=AGE_GROUPS, columns=['actual_minus_pre', 'post_minus_pre_future', 'post_minus_pre_present'])

    forecast_type = 'group'
    data_type = 'male'

    for age_group in AGE_GROUPS:
        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'key_nb.csv')
        df.loc[age_group] = pd.read_csv(path, index_col=0, header=None).T.values

    temp = df.sum(axis=0)

    df = pd.DataFrame(index= AGE_GROUPS, columns=['actual_minus_pre', 'post_minus_pre_future', 'post_minus_pre_present'])

    data_type = 'female'

    for age_group in AGE_GROUPS:
        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'key_nb.csv')
        df.loc[age_group] = pd.read_csv(path, index_col=0, header=None).T.values

    summary_df.loc['age_gender'] = temp + df.sum(axis=0)

    summary_df.to_csv(os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'summary_' + start_date + '.csv'))


def gen_reg_analysis(params, output_path, clean_data_path):
    analysis_date = params['analysis_date']

    df = pd.DataFrame(columns=['unemp_type', 'gender_group', 'age_group', 'date_start', 'adjusted_rsquared', 'frac_outliers'])

    age_groups_ext = list(AGE_GROUPS)
    age_groups_ext.append('total')

    i = 0
    for unemp_type in UNEMP_TYPES:
        for data_type in DATA_TYPES:
            for age_group in age_groups_ext:
                for date_start in DATES_START:
                    path = os.path.join(output_path, analysis_date, 'model', unemp_type, data_type, age_group, date_start)
                    full_path = os.path.join(path, 'regression_result.pickle')
                    res = sm.load_pickle(full_path)
                    frac_outliers = (res.outlier_test()['bonf(p)'] < 0.05).sum() / res.nobs

                    data_vec = (unemp_type, data_type, age_group, date_start, res.rsquared_adj, frac_outliers)
                    df.loc[i] = data_vec
                    i += 1

    data = df.groupby('unemp_type').adjusted_rsquared.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    title = r'$\text{Average Adjusted }R^{2}\text{ by Unemployment Type}$'
    fig.update_layout(title_text=title)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'regression_analysis', 'rsquared_by_unemployment.pdf')
    fig.write_image(path, format='pdf')

    data = df.groupby('unemp_type').frac_outliers.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    title = 'Average Fraction of Outliers by Unemployment Type'
    fig.update_layout(title_text=title)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'regression_analysis', 'frac_outliers_by_unemployment.pdf')
    fig.write_image(path, format='pdf')

    data = df.groupby('gender_group').adjusted_rsquared.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    title = r'$\text{Average Adjusted }R^{2}\text{ by Gender Group}$'
    fig.update_layout(title_text=title)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'regression_analysis', 'rsquared_by_gender.pdf')
    fig.write_image(path, format='pdf')

    data = df.groupby('gender_group').frac_outliers.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    title = 'Average Fraction of Outliers by Gender Group'
    fig.update_layout(title_text=title)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'regression_analysis', 'frac_outliers_by_gender.pdf')
    fig.write_image(path, format='pdf')

    data = df.groupby('age_group').adjusted_rsquared.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    title = r'$\text{Average Adjusted }R^{2}\text{ by Age Group}$'
    fig.update_layout(title_text=title)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'regression_analysis', 'rsquared_by_age.pdf')
    fig.write_image(path, format='pdf')

    data = df.groupby('age_group').frac_outliers.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    title = 'Average Fraction of Outliers by Age Group'
    fig.update_layout(title_text=title)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'regression_analysis', 'frac_outliers_by_age.pdf')
    fig.write_image(path, format='pdf')

    data = df.groupby('date_start').adjusted_rsquared.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    title = r'$\text{Average Adjusted }R^{2}\text{ by Start Date}$'
    fig.update_layout(title_text=title)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'regression_analysis', 'rsquared_by_start_date.pdf')
    fig.write_image(path, format='pdf')

    data = df.groupby('date_start').frac_outliers.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data))
    title = r'Average Fraction of Outliers by Start Date'
    fig.update_layout(title_text=title)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'regression_analysis', 'frac_outliers_by_start_date.pdf')
    fig.write_image(path, format='pdf')


def summary_figures(params, output_path, clean_data_path):
    analysis_date = params['analysis_date']

    df_suicide_monthly, _ = load_suicide_data(params, clean_data_path)
    suicide = df_suicide_monthly.total.total.rename('Total')
    suicide.index = pd.to_datetime(suicide.index)

    # Aggregate
    forecast_type = 'aggregate'
    data_type = 'total'
    age_group = 'total'
    start_date = '2010-01'

    path_start = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date)
    pre_path = os.path.join(path_start, 'pre_preds.csv')
    pre_pred = pd.read_csv(pre_path, index_col=0).pred
    pre_pred.index = pd.to_datetime(pre_pred.index).rename('date')

    post_path = os.path.join(path_start, 'post_preds.csv')
    post_pred = pd.read_csv(post_path, index_col=0).pred
    post_pred.index = pd.to_datetime(post_pred.index).rename('date')

    conf_int_path = os.path.join(path_start, 'pre_conf_int.csv')
    pre_conf_int = pd.read_csv(conf_int_path, index_col=0).values

    preds = (pre_pred, post_pred, pre_conf_int)

    date_end = suicide.index[-1]
    plot_start = '2019-01'
    args = filter_dates(suicide, preds, plot_start, date_end)
    fig = plot_preds_ts(*args)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'aggregate', 'unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_induced_suicides_ts(args[0], args[2])
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'aggregate', 'induced_suicides_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_explained_unemp_ts(args[0], args[1], suicide.name)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'aggregate', 'explained_unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'aggregate', 'pre_preds.csv')
    pre_pred.to_csv(path)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'aggregate', 'post_preds.csv')
    post_pred.to_csv(path)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'aggregate', 'suicide.csv')
    suicide.to_csv(path)

    # Age Only
    pre_pred[:] = 0.
    post_pred[:] = 0.
    pre_conf_int[:] = 0.

    forecast_type = 'group'
    data_type = 'total'

    for age_group in AGE_GROUPS:
        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'pre_preds.csv')
        pre_pred += pd.read_csv(path, index_col=0).pred.values

        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'post_preds.csv')
        post_pred += pd.read_csv(path, index_col=0).pred.values

        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'pre_conf_int.csv')
        pre_conf_int += pd.read_csv(path, index_col=0).values

    args = filter_dates(suicide, preds, plot_start, date_end)
    fig = plot_preds_ts(*args)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_only', 'unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_induced_suicides_ts(args[0], args[2])
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_only', 'induced_suicides_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_explained_unemp_ts(args[0], args[1], suicide.name)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_only', 'explained_unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_only', 'pre_preds.csv')
    pre_pred.to_csv(path)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_only', 'post_preds.csv')
    post_pred.to_csv(path)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_only', 'suicide.csv')
    suicide.to_csv(path)

    # Gender only
    pre_pred[:] = 0.
    post_pred[:] = 0.
    pre_conf_int[:] = 0.

    forecast_type = 'group'
    data_type = 'male'
    age_group = 'total'
    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'pre_preds.csv')
    pre_pred += pd.read_csv(path, index_col=0).pred.values

    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'post_preds.csv')
    post_pred += pd.read_csv(path, index_col=0).pred.values

    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'pre_conf_int.csv')
    pre_conf_int += pd.read_csv(path, index_col=0).values

    data_type = 'female'
    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'pre_preds.csv')
    pre_pred += pd.read_csv(path, index_col=0).pred.values

    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'post_preds.csv')
    post_pred += pd.read_csv(path, index_col=0).pred.values

    path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                        data_type, age_group, start_date, 'pre_conf_int.csv')
    pre_conf_int += pd.read_csv(path, index_col=0).values

    args = filter_dates(suicide, preds, plot_start, date_end)
    fig = plot_preds_ts(*args)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'gender_only', 'unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_induced_suicides_ts(args[0], args[2])
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'gender_only', 'induced_suicides_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_explained_unemp_ts(args[0], args[1], suicide.name)
    full_path =  os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'gender_only', 'explained_unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'gender_only', 'pre_preds.csv')
    pre_pred.to_csv(path)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'gender_only', 'post_preds.csv')
    post_pred.to_csv(path)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'gender_only', 'suicide.csv')
    suicide.to_csv(path)

    # Age-gender
    pre_pred[:] = 0.
    post_pred[:] = 0.
    pre_conf_int[:] = 0.

    forecast_type = 'group'
    data_type = 'male'

    for age_group in AGE_GROUPS:
        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'pre_preds.csv')
        pre_pred += pd.read_csv(path, index_col=0).pred.values

        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'post_preds.csv')
        post_pred += pd.read_csv(path, index_col=0).pred.values

        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'pre_conf_int.csv')
        pre_conf_int += pd.read_csv(path, index_col=0).values

    data_type = 'female'

    for age_group in AGE_GROUPS:
        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'pre_preds.csv')
        pre_pred += pd.read_csv(path, index_col=0).pred.values

        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'post_preds.csv')
        post_pred += pd.read_csv(path, index_col=0).pred.values

        path = os.path.join(output_path, analysis_date, 'model', forecast_type,
                            data_type, age_group, start_date, 'pre_conf_int.csv')
        pre_conf_int += pd.read_csv(path, index_col=0).values

    args = filter_dates(suicide, preds, plot_start, date_end)
    fig = plot_preds_ts(*args)
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_gender', 'unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_induced_suicides_ts(args[0], args[2])
    full_path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_gender', 'induced_suicides_ts.pdf')
    fig.write_image(full_path, format='pdf')

    fig = plot_explained_unemp_ts(args[0], args[1], suicide.name)
    full_path =  os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_gender', 'explained_unemp_ts.pdf')
    fig.write_image(full_path, format='pdf')

    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_gender', 'pre_preds.csv')
    pre_pred.to_csv(path)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_gender', 'post_preds.csv')
    post_pred.to_csv(path)
    path = os.path.join(output_path, analysis_date, 'result_analysis', 'summary', 'age_gender', 'suicide.csv')
    suicide.to_csv(path)



def analyze_results(params, output_path, clean_data_path):
    # summary_table(params, output_path)
    # gen_figs(params, output_path, clean_data_path)
    # gen_reg_analysis(params, output_path, clean_data_path)
    summary_figures(params, output_path, clean_data_path)


if __name__ == '__main__':
    import yaml

    params_path = os.path.join(os.pardir, 'parameters.yml')
    output_path = os.path.join(os.pardir, 'output')
    clean_data_path = os.path.join(os.pardir, 'clean_data')

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    analyze_results(params, output_path, clean_data_path)
