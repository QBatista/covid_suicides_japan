"""
A script to clean the life expectancy data.

"""

import os
import numpy as np
import pandas as pd


AGE_GROUPS = ['0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79',
              '80_99']
MALE_SHEET_NAME = '令和元年簡易生命表（男）'
FEMALE_SHEET_NAME = '令和元年簡易生命表（女）'


def get_series(df):
    age_1 = df.iloc[13:63, 1]
    le_1 = df.iloc[13:63, 13]
    age_2 = df.iloc[5:55, 16]
    le_2 = df.iloc[5:55, 28]

    srs_1 = pd.Series(le_1.values, index=age_1.values)
    srs_2 = pd.Series(le_2.values, index=age_2.values)

    srs = srs_1.append(srs_2)

    return srs


def life_expectancy(params, load_path, save_path):
    """
    Clean the life expectancy data located at `load_path` and save it
    to `save_path` based on `params['analysis_date']`.

    """

    # Unpack arguments
    analysis_date = params['analysis_date']

    # Get full paths
    load_path = os.path.join(load_path, analysis_date, 'life_expectancy.xlsx')
    save_path = os.path.join(save_path, analysis_date, 'life_expectancy.csv')

    # Load data
    xls = pd.read_excel(load_path, None, header=None)
    data = {'male': get_series(xls[MALE_SHEET_NAME]),
            'female': get_series(xls[FEMALE_SHEET_NAME])}
    df = pd.DataFrame(data)

    # Initialize cleaned dataframe
    df_final = pd.DataFrame(columns=['male', 'female'], index=AGE_GROUPS)

    # Compute average life expectancy for each age group
    for a in AGE_GROUPS:
        df_final.loc[a, :] = df.loc[slice(*map(int, a.split('_')))].mean()

    # Rename index
    df_final.index.rename('age_group', inplace=True)

    # Save
    df_final.to_csv(save_path)


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    save_path = '../clean_data/'
    load_path = '../raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    life_expectancy(params, load_path, save_path)
