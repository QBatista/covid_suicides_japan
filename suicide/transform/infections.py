"""
A script to clean the infections data.

"""

import os
import numpy as np
import pandas as pd
import datefinder


GROUPS = ['0_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79',
          '80_89', '90_99', 'subtotal', 'old', 'undisclosed', 'total']
AGE_GROUPS = GROUPS[:-4]
OLD_GROUPS = ['60_69', '70_79', '80_89', '90_99']
GENDERS = ['male', 'female', 'undisclosed']
OUTPUT_GROUPS = ['0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79',
                 '80_89', '90_99', 'total']
α = 1  # Laplace smoothing


def attrib_undisclosed_gender(df_month):
    total = df_month['male'] + df_month['female'] + 2 * α

    # Compute weights
    ω = (df_month['male'] + α) / total

    # Increment counts
    df_month['male'] += ω.fillna(0) * df_month['undisclosed']
    df_month['female'] += (1 - ω).fillna(0) * df_month['undisclosed']


def attrib_undisclosed_age(df_month):
    # Male undisclosed age groups
    age_total = df_month.male.loc[:, AGE_GROUPS].sum(axis=1)

    # Compute weights
    ω = df_month.male.loc[:, AGE_GROUPS].div(age_total, axis=0).fillna(0)

    # Increment
    mask = [('male', a) for a in AGE_GROUPS]
    counts = df_month.male.undisclosed
    df_month.loc[:, mask] += ω.multiply(counts, axis=0).values

    # Female undisclosed age groups
    age_total = df_month.female.loc[:, AGE_GROUPS].sum(axis=1)

    # Compute weights
    ω = df_month.female.loc[:, AGE_GROUPS].div(age_total, axis=0).fillna(0)

    # Increment
    mask = [('female', a) for a in AGE_GROUPS]
    counts = df_month.female.undisclosed
    df_month.loc[:, mask] += ω.multiply(counts, axis=0).values


def attrib_old(df_month):
    # Attribute male old counts
    mask = [('male', a) for a in OLD_GROUPS]
    old_total = df_month.loc[:, mask].sum(axis=1)

    # Compute weights
    ω = df_month.loc[:, mask].div(old_total, axis=0)

    # Increment
    df_month.loc[:, mask] += ω.multiply(df_month.male.old, axis=0).fillna(0)

    # Attribute female old counts
    mask = [('female', a) for a in OLD_GROUPS]
    old_total = df_month.loc[:, mask].sum(axis=1)

    # Compute weights
    ω = df_month.loc[:, mask].div(old_total, axis=0)
    df_month.loc[:, mask] += ω.multiply(df_month.female.old, axis=0).fillna(0)


def infections(params, load_path, save_path):
    """
    Clean the infections distribution data located at `load_path` and save it
    to `save_path` based on `params['analysis_date']`.

    """

    # Unpack arguments
    analysis_date = params['analysis_date']

    # Get full paths
    load_path = os.path.join(load_path, analysis_date, 'infections.xlsx')
    save_path = os.path.join(save_path, analysis_date, 'infection_deaths.csv')

    # Load raw data
    file = pd.read_excel(load_path, None, header=None)

    # Initialize dataframe
    cols = ((g, a) for g in GENDERS for a in GROUPS)
    df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(cols))

    # Get data for each sheet in the Excel file. The data contains cumulative
    # counts at a weekly frequency (for the most part). To aggregate it to
    # a monthly frequency, it is convenient to first resample it a daily
    # frequency. For a given period, we attribute deaths evenly among
    # individual days.
    ts_prev = pd.to_datetime('2020/02/29')  # Initial time stamp
    df.loc[ts_prev, :] = 0
    cache = np.zeros((3, 14))  # Cache previous data for different genders

    # Iterate backwards from oldest to newest cumulative count
    for key in list(file.keys())[::-1]:
        # Find timestamp (note that the name of the sheet is unreliable)
        matches = list(datefinder.find_dates(file[key].iloc[0, 0]))
        ts_curr = pd.to_datetime(matches[0])

        # Get the number of days between current and previous time stamp
        time_δ = (ts_curr - ts_prev).days

        # Store **daily** data for each gender resampled
        for i, g in enumerate(GENDERS):
            val = file[key].iloc[4:18, i+1].values
            df.loc[ts_curr, (g, )] = (val - cache[i]) / time_δ

            # Store previous data for gender `g`
            cache[i] = val

        # Update timestamp
        ts_prev = ts_curr

    # Fill days between time stamps
    df_days = df.resample('D').bfill()

    # Aggregate to monthly frequency
    df_month = df_days.resample('M').sum()

    # Attribute undisclosed gender infection deaths
    attrib_undisclosed_gender(df_month)

    # Attribute undisclosed age infection deaths
    attrib_undisclosed_age(df_month)

    # Attribute old infection deaths
    attrib_old(df_month)

    # Drop some columns and first row
    age_cols = [(g, a) for g in ['male', 'female'] for a in AGE_GROUPS]
    df_month = df_month.loc[:, age_cols]
    df_month = df_month.iloc[1:]
    df_month = df_month.to_period("M").to_timestamp()

    # Compute totals
    df_month.loc[:, ('male', 'total')] = df_month.male.sum(axis=1)
    df_month.loc[:, ('female', 'total')] = df_month.female.sum(axis=1)

    # Aggregate some age groups
    c = ('male', '0_19')
    df_month.loc[:, c] = df_month.male[['0_9', '10_19']].sum(axis=1)

    c = ('female', '0_19')
    df_month.loc[:, c] = df_month.female[['0_9', '10_19']].sum(axis=1)

    # Select output groups
    output_cols = [(g, a) for g in ['male', 'female'] for a in OUTPUT_GROUPS]
    df_final = df_month.loc[:, output_cols]

    # Melt
    var_name = ['gender_group', 'age_group']
    df_final = df_final.melt(var_name=var_name, ignore_index=False)

    # Re-order columns
    df_final = df_final[['age_group', 'gender_group', 'value']]
    df_final.index.rename('date', inplace=True)

    # Save
    df_final.to_csv(save_path)


if __name__ == '__main__':
    import yaml

    params_path = os.path.join(os.pardir, 'parameters.yml')
    save_path = os.path.join(os.pardir, 'clean_data')
    load_path = os.path.join(os.pardir, 'raw_data')

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    infections(params, load_path, save_path)
