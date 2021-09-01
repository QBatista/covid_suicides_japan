"""
A script to clean the raw annual suicide distribution data.

"""

import os
import pandas as pd
import numpy as np


COLUMNS = ['age_group', 'gender_group', 'value']
AGE_GROUPS = ['total', '0_19', '20_29', '30_39', '40_49', '50_59', '60_69',
              '70_79', '80_99']


def suicide_dist_annual(params, load_path, save_path):

    analysis_date = params['analysis_date']

    load_path = os.path.join(load_path, analysis_date, 'suicide_dist', 'annual')
    save_path = os.path.join(save_path, analysis_date, 'suicide_dist_annual.csv')

    columns = pd.MultiIndex.from_product([('male', 'female', 'total'), AGE_GROUPS])
    data=pd.DataFrame(columns=columns)
    for year in range(2009,2021):

        filename = str(year)+".xls"
        path = os.path.join(load_path, filename)
        ts = pd.to_datetime('%i/%i/%i' % (year, 1, 1))
        xls = pd.read_excel(path,header=None)
        male_data = xls.iloc[5, 4:41:4].values.copy()
        male_data[1:9] += male_data[1:9] * male_data[9] / (male_data[0] - male_data[9])

        female_data = xls.iloc[6, 4:41:4].values.copy()
        female_data[1:9] += female_data[1:9] * female_data[9] / (female_data[0] - female_data[9])

        row = np.r_[male_data[:9], female_data[:9], male_data[:9] + female_data[:9]]
        data.loc[ts,:] = row

    data = data.melt(var_name=['gender_group', 'age_group'], ignore_index=False)
    data.index.rename('date', inplace=True)
    data.to_csv(save_path)


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    save_path = '../clean_data/'
    load_path = '../raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_dist_annual(params, load_path, save_path)
