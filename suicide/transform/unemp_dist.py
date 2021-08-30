"""
A script to clean the unemployment data.

"""

import os
import pandas as pd
from statsmodels.tsa.seasonal import STL


GENDER_GROUPS = ('total', 'male', 'female')
ORIGINAL_AGE_GROUPS = ('total', '15_24', '25_34', '35_44', '45_54', '55_64',
                       '65_70')
OUTPUT_AGE_GROUPS = ('total', '0_19', '20_29', '30_39', '40_49', '50_59',
                     '60_69', '70_79', '80_99')


def remove_seasonality(x):
    res = STL(x).fit()

    return res.trend + res.resid


def unemp_dist(params, load_path, save_path):
    # Unpack arguments
    analysis_date = params['analysis_date']

    file_path = os.path.join(load_path, analysis_date, 'unemp_dist.xlsx')
    # Load raw data
    data = pd.read_excel(file_path, sheet_name='原数値', header=None,
                         skiprows=10, skipfooter=4)

    year = pd.to_numeric(data.iloc[:, 0].shift(-1),
                     errors='coerce').fillna(method='ffill').astype(int)
    month = data.iloc[:, 1].str.extract('(\d+)').iloc[:, 0]
    date = pd.to_datetime({'year': year, 'month': month, 'day': 1})

    cols = pd.MultiIndex.from_product((GENDER_GROUPS, ORIGINAL_AGE_GROUPS))
    df = pd.DataFrame(columns=cols, index=date)

    df.loc[:, ('total', 'total')] = pd.to_numeric(data.iloc[:, 4], errors='coerce').values
    df.loc[:, (('total', a) for a in ORIGINAL_AGE_GROUPS[1:])] = data.iloc[:, 6:12].apply(lambda x: pd.to_numeric(x, errors='coerce')).values

    df.loc[:, ('male', 'total')] = pd.to_numeric(data.iloc[:, 13], errors='coerce').values
    df.loc[:, (('male', a) for a in ORIGINAL_AGE_GROUPS[1:])] = data.iloc[:, 15:21].apply(lambda x: pd.to_numeric(x, errors='coerce')).values

    df.loc[:, ('female', 'total')] = pd.to_numeric(data.iloc[:, 22], errors='coerce').values
    df.loc[:, (('female', a) for a in ORIGINAL_AGE_GROUPS[1:])] = data.iloc[:, 24:30].apply(lambda x: pd.to_numeric(x, errors='coerce')).values

    df.dropna(inplace=True)

    cols = pd.MultiIndex.from_product((GENDER_GROUPS, OUTPUT_AGE_GROUPS))
    df_final = pd.DataFrame(columns=cols, index=df.index)

    n_output = len(OUTPUT_AGE_GROUPS)

    for k in range(len(GENDER_GROUPS)):
        for i in range(len(ORIGINAL_AGE_GROUPS) - 2):
            col_ind = 2 + i + k * n_output
            df_final.iloc[:,col_ind] = df[GENDER_GROUPS[k]].iloc[:, 1+i:3+i].mean(axis=1)

    for g in GENDER_GROUPS:
        df_final.loc[:, (g, 'total')] = df.loc[:, (g, 'total')]
        df_final.loc[:, (g, '0_19')] = df.loc[:, (g, '15_24')] / 4
        df_final.loc[:, (g, '70_79')] = 0.
        df_final.loc[:, (g, '80_99')] = 0.

    df_final = df_final['2008':].apply(remove_seasonality)
    df_final.index.rename('date', inplace=True)
    df_final = df_final.melt(var_name=['gender_group', 'age_group'], ignore_index=False)

    path = os.path.join(save_path, analysis_date, 'unemp_dist.csv')
    df_final.to_csv(path)


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    save_path = '../clean_data/'
    load_path = '../raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    unemp_dist(params, load_path, save_path)
