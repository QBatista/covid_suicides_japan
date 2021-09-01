"""
A script to clean the raw monthly suicide distribution.

"""

import os
import pandas as pd
import numpy as np


COLUMNS = ['age_group', 'gender_group', 'value']
AGE_GROUPS = ['total', '0_19', '20_29', '30_39', '40_49', '50_59', '60_69',
              '70_79', '80_99']


def suicide_dist_monthly(params, load_path, save_path):
    # Unpack arguments
    analysis_date = params['analysis_date']

    annual_path = os.path.join(save_path, analysis_date, 'suicide_dist_annual.csv')
    load_path = os.path.join(load_path, analysis_date, 'suicide_dist', 'monthly')
    save_path = os.path.join(save_path, analysis_date, 'suicide_dist_monthly.csv')

    columns = pd.MultiIndex.from_product([('male', 'female', 'total'), AGE_GROUPS])
    data=pd.DataFrame(columns=columns)
    for year in range(2009,2022):
        for month in range(1,13):
            if year==2021 and month>=7:
                break

            path = os.path.join(load_path, str(year)+"-"+f"{month:02d}"+".xls")
            ts = pd.to_datetime('%i/%i/%i' % (year, month, 1))
            xls = pd.read_excel(path,header=None)
            male_data = xls.iloc[5, 4:41:4].values.copy()
            male_data[1:9] += male_data[1:9] * male_data[9] / (male_data[0] - male_data[9])

            female_data = xls.iloc[6, 4:41:4].values.copy()
            female_data[1:9] += female_data[1:9] * female_data[9] / (female_data[0] - female_data[9])

            row = np.r_[male_data[:9], female_data[:9], male_data[:9] + female_data[:9]]
            data.loc[ts,:] = row

    #data.melt().to_csv('suicide_dist.csv')
    suicide_annual = pd.read_csv(annual_path, index_col=0)
    suicide_annual = suicide_annual.pivot(columns=['gender_group', 'age_group'], values='value')
    suicide_annual.index = pd.to_datetime(suicide_annual.index)

    suicide_monthly = data[suicide_annual.columns]
    suicide_monthly.index = pd.to_datetime(suicide_monthly.index)

    last_year = str(suicide_annual.index[-1].year)
    suicide_monthly_filt = suicide_monthly[:last_year]
    grp = suicide_monthly_filt.groupby(suicide_monthly_filt.index.year)
    suicide_monthly_agg = grp.sum()

    diff = suicide_annual - suicide_monthly_agg.values

    diff_monthly = pd.DataFrame().reindex_like(suicide_monthly_filt)
    diff_monthly.loc[diff.index, :] = diff
    diff_monthly.pad(inplace=True)

    weights = grp.apply(lambda x: x / x.sum())
    suicide_monthly = suicide_monthly.add(weights * diff_monthly, fill_value=0.)

    suicide_monthly = suicide_monthly.melt(var_name=['gender_group', 'age_group'], ignore_index=False)
    data.index.rename('date', inplace=True)
    suicide_monthly.to_csv(save_path)


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    save_path = '../clean_data/'
    load_path = '../raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_dist_monthly(params, load_path, save_path)
