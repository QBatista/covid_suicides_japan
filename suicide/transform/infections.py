"""
A script to clean the infections data.

"""

import os
import numpy as np
import pandas as pd
import datefinder


GROUPS = ['0_9', '10_19', '20_29', '30_39', '40_49', '50_59',
          '60_69', '70_79', '80_89', '90_99', 'subtotal', 'old',
          'undisclosed', 'total']
AGE_GROUPS = GROUPS[:-4]
OLD_GROUPS = ['60_69', '70_79', '80_89', '90_99']
GENDERS = ['male', 'female', 'undisclosed']
OUTPUT_GROUPS = ['0_19','20_29', '30_39', '40_49', '50_59',
          '60_69', '70_79', '80_99', 'total']
α = 1  # Laplace smoothing


def infections(params, load_path, save_path):
    # Unpack arguments
    analysis_date = params['analysis_date']

    load_path = os.path.join(load_path, analysis_date, 'infections.xlsx')
    save_path = os.path.join(save_path, analysis_date, 'infection_deaths.csv')

    cols = ((g, a) for g in GENDERS for a in GROUPS)
    df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(cols))

    file = pd.read_excel(load_path, None, header=None)

    ts_prev = pd.to_datetime('2020/02/29')
    df.loc[pd.to_datetime('2020/02/29'), :] = 0
    last_val = np.zeros((3, 14))

    for key in list(file.keys())[::-1]:
        matches = list(datefinder.find_dates(file[key].iloc[0, 0]))
        ts_curr = pd.to_datetime(matches[0])
        time_δ = (ts_curr - ts_prev).days

        for i, g in enumerate(GENDERS):
            val = file[key].iloc[4:18, i+1].values
            df.loc[ts_curr, (g, )] = (val - last_val[i]) / time_δ
            last_val[i] = val

        ts_prev = ts_curr

    df_days = df.resample('D').bfill()
    df_month = df_days.resample('M').sum()

    total = df_month['male'] + df_month['female'] + 2 * α
    df_month['male'] += ((df_month['male'] + α) / total).fillna(0) * df_month['undisclosed']
    df_month['female'] += ((df_month['female'] + α) / total).fillna(0) * df_month['undisclosed']

    age_total = df_month.male.loc[:, AGE_GROUPS].sum(axis=1)
    df_month.loc[:, [('male', a) for a in AGE_GROUPS]] += \
    df_month.male.loc[:, AGE_GROUPS].div(age_total, axis=0).fillna(0).multiply(df_month.male.undisclosed, axis=0).values

    age_total = df_month.female.loc[:, AGE_GROUPS].sum(axis=1)
    df_month.loc[:, [('female', a) for a in AGE_GROUPS]] += \
    df_month.female.loc[:, AGE_GROUPS].div(age_total, axis=0).fillna(0).multiply(df_month.female.undisclosed, axis=0).values

    old_total = df_month.loc[:, [('male', a) for a in OLD_GROUPS]].sum(axis=1)
    df_month.loc[:, [('male', a) for a in OLD_GROUPS]] += \
    df_month.loc[:, [('male', a) for a in OLD_GROUPS]].div(old_total, axis=0).multiply(df_month.male.old, axis=0).fillna(0)

    old_total = df_month.loc[:, [('female', a) for a in OLD_GROUPS]].sum(axis=1)
    df_month.loc[:, [('female', a) for a in OLD_GROUPS]] += \
    df_month.loc[:, [('female', a) for a in OLD_GROUPS]].div(old_total, axis=0).multiply(df_month.female.old, axis=0).fillna(0)

    df_final = df_month.loc[:, [(g, a) for g in ['male', 'female'] for a in AGE_GROUPS]]
    df_final = df_final.iloc[1:]
    df_final = df_final.to_period("M").to_timestamp()

    df_final.loc[:, ('male', 'total')] = df_final.male.sum(axis=1)
    df_final.loc[:, ('female', 'total')] = df_final.female.sum(axis=1)

    df_final.loc[:, ('male', '0_19')] = df_final.male['0_9'] + df_final.male['10_19']
    df_final.loc[:, ('male', '80_99')] = df_final.male['80_89'] + df_final.male['90_99']
    df_final.loc[:, ('female', '0_19')] = df_final.female['0_9'] + df_final.female['10_19']
    df_final.loc[:, ('female', '80_99')] = df_final.female['80_89'] + df_final.female['90_99']

    df_final = df_final.loc[:, [(g, a) for g in ['male', 'female'] for a in OUTPUT_GROUPS]]
    df_final = df_final.melt(var_name=['gender_group', 'age_group'], ignore_index=False)
    df_final = df_final[['age_group', 'gender_group', 'value']]
    df_final.index.rename('date', inplace=True)
    df_final.to_csv(save_path)


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    save_path = '../clean_data/'
    load_path = '../raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    infections(params, load_path, save_path)
