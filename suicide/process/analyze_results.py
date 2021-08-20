

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATES_START = ('2009-01',
               '2010-01',
               '2011-01',
               '2012-01')
UNEMP_TYPES = ('aggregate', 'group')
AGE_GROUPS = ('0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79',
              '80_99')


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


def gen_figs(params, output_path, clean_data_path):
    analysis_date = params['analysis_date']

    for date_start in DATES_START:
        for unemp_type in UNEMP_TYPES:
            df_life_exp, df_infections, df_suicide_m, df_suicide_f = load_data(params, date_start, unemp_type, output_path, clean_data_path)

            suicide_m_le = df_suicide_m.multiply(df_life_exp['male'])
            suicide_m_sum = suicide_m_le.sum(axis=0).rename('Male')

            suicide_f_le = df_suicide_f.multiply(df_life_exp['female'])
            suicide_f_sum = suicide_f_le.sum(axis=0).rename('Female')

            suicide_mf_le = suicide_m_le + suicide_f_le
            suicide_le_total = (suicide_m_sum + suicide_f_sum).rename('Total')

            mask = df_infections.group.isin(AGE_GROUPS)
            infections_data = df_infections[mask].pivot(columns=['sex', 'group'], values='infection_death')

            infections_m_le = infections_data['male'].multiply(df_life_exp['male'])
            infections_f_le = infections_data['female'].multiply(df_life_exp['female'])

            infections_m_le_sum = infections_m_le.sum(axis=0).rename('Male')
            infections_f_le_sum = infections_f_le.sum(axis=0).rename('Female')

            infections_mf_le = infections_m_le + infections_f_le

            infections_le_total = (infections_m_le_sum + infections_f_le_sum).rename('Total')

            # Lost life expectancy by age groups
            title = suicide_le_total.name + ' Years of Life Expectancy Lost Due to Covid Induced Suicides vs. Infections by Age Group'
            suicide_le_total.index = [group.replace('_', ' to ') for group in suicide_le_total.index]
            fig = plot_loss_life_exp(suicide_le_total, infections_le_total, title)
            full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'total', date_start, 'lost_life_exp_by_age.pdf')
            fig.write_image(full_path, format='pdf')

            title = suicide_m_sum.name + ' Years of Life Expectancy Lost Due to Covid Induced Suicides vs. Infections by Age Group'
            suicide_m_sum.index = [group.replace('_', ' to ') for group in suicide_m_sum.index]
            fig = plot_loss_life_exp(suicide_m_sum, infections_m_le_sum, title)
            full_path = os.path.join(output_path, analysis_date, 'result_analysis', unemp_type, 'male', date_start, 'lost_life_exp_by_age.pdf')
            fig.write_image(full_path, format='pdf')

            title = suicide_f_sum.name + ' Years of Life Expectancy Lost Due to Covid Induced Suicides vs. Infections by Age Group'
            suicide_f_sum.index = [group.replace('_', ' to ') for group in suicide_f_sum.index]
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
    df = pd.DataFrame(index= AGE_GROUPS, columns=['actual_minus_pre', 'post_minus_pre_future', 'post_minus_pre_present'])

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

    summary_df.to_csv(os.path.join(output_path, analysis_date, 'result_analysis/summary_' + start_date + '.csv'))


def analyze_results(params, output_path, clean_data_path):
    summary_table(params, output_path)
    gen_figs(params, output_path, clean_data_path)


if __name__ == '__main__':
    import yaml

    params_path = os.path.join(os.pardir, 'parameters.yml')
    output_path = os.path.join(os.pardir, 'output')
    clean_data_path = os.path.join(os.pardir, 'clean_data')

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    analyze_results(params, output_path, clean_data_path)
