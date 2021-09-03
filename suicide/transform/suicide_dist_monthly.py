"""
A script to clean the raw monthly suicide distribution.

"""

import os
import pandas as pd
import numpy as np


COLUMNS = ['age_group', 'gender_group', 'value']
AGE_GROUPS = ['total', '0_19', '20_29', '30_39', '40_49', '50_59', '60_69',
              '70_79', '80_99']
GENDER_GROUPS = ['male', 'female', 'total']


def get_data(load_path):
    # Initialize dataframe
    columns = pd.MultiIndex.from_product([GENDER_GROUPS, AGE_GROUPS])
    data = pd.DataFrame(columns=columns)

    for filename in os.listdir(load_path):
        year = int(filename[:4])
        month = int(filename[5:7])

        # Get excel file
        path = os.path.join(load_path, filename)
        ts = pd.to_datetime('%i/%i/%i' % (year, month, 1))
        xls = pd.read_excel(path, header=None)

        # Extract male data
        male_data = xls.iloc[5, 4:41:4].values.copy()

        # Adjust male data
        ω = male_data[9] / (male_data[0] - male_data[9])
        male_data[1:9] += male_data[1:9] * ω

        # Extract female data
        female_data = xls.iloc[6, 4:41:4].values.copy()

        # Adjust female data
        ω = female_data[9] / (female_data[0] - female_data[9])
        female_data[1:9] += female_data[1:9] * ω

        # Update dataframe
        total = male_data[:9] + female_data[:9]
        row = np.r_[male_data[:9], female_data[:9], total]
        data.loc[ts, :] = row

    return data


def adjust_annual(suicide_annual, suicide_monthly):
    # Compute annual values by aggregating monthly data
    last_year = str(suicide_annual.index[-1].year)
    suicide_monthly_filt = suicide_monthly.loc[:last_year]
    grp = suicide_monthly_filt.groupby(suicide_monthly_filt.index.year)
    suicide_monthly_agg = grp.sum()

    # Compute difference between annual and monthly aggregated
    diff = suicide_annual - suicide_monthly_agg.values

    # Increment difference for each month
    diff_monthly = pd.DataFrame().reindex_like(suicide_monthly_filt)
    diff_monthly.loc[diff.index, :] = diff
    diff_monthly.pad(inplace=True)
    ω = grp.apply(lambda x: x / x.sum())
    suicide_monthly = suicide_monthly.add(ω * diff_monthly, fill_value=0.)

    return suicide_monthly


def suicide_dist_monthly(params, load_path, save_path):
    """
    Clean the monthly suicide distribution data located at `load_path` and save
    it to `save_path` based on `params['analysis_date']`.

    """

    # Unpack arguments
    analysis_date = params['analysis_date']

    # Update paths
    annual_path = os.path.join(save_path, analysis_date,
                               'suicide_dist_annual.csv')
    load_path = os.path.join(load_path, analysis_date, 'suicide_dist',
                             'monthly')
    save_path = os.path.join(save_path, analysis_date,
                             'suicide_dist_monthly.csv')

    # Get monthly data
    data = get_data(load_path)

    # Load annual data
    suicide_annual = pd.read_csv(annual_path, index_col=0)
    var_name = ['gender_group', 'age_group']
    suicide_annual = suicide_annual.pivot(columns=var_name, values='value')
    suicide_annual.index = pd.to_datetime(suicide_annual.index)

    # Adjust based on annual data
    suicide_monthly = data[suicide_annual.columns]
    suicide_monthly.index = pd.to_datetime(suicide_monthly.index)
    suicide_monthly.sort_index(inplace=True)
    suicide_monthly = adjust_annual(suicide_annual, suicide_monthly)

    # Melt
    suicide_monthly = suicide_monthly.melt(var_name=var_name,
                                           ignore_index=False)

    # Rename index
    suicide_monthly.index.rename('date', inplace=True)

    # Save
    suicide_monthly.to_csv(save_path)


if __name__ == '__main__':
    import yaml

    params_path = os.path.join(os.pardir, 'parameters.yml')
    save_path = os.path.join(os.pardir, 'clean_data')
    load_path = os.path.join(os.pardir, 'raw_data')

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_dist_monthly(params, load_path, save_path)
