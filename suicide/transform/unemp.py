"""
A script to clean the unemployment data.

"""

import pandas as pd


def unemp(params, load_path, save_path):
    # Unpack arguments
    analysis_date = params['analysis_date']

    load_path += analysis_date + '/'
    save_path += analysis_date + '/'

    # Load raw data
    df = pd.read_excel(load_path + 'unemp.xlxs', skiprows=8)

    # Extract relevant columns
    df_unemp = df[['Both sexes.5', 'Male.5', 'Female.5']].copy()

    # Rename columns
    df_unemp.columns = ['total', 'male', 'female']

    # Construct time index
    year = pd.to_numeric(df['Unnamed: 0'].shift(-1),
                         errors='coerce').fillna(method='ffill')
    month = df['Unnamed: 1'].str.extract('(\d+)')

    df_unemp.loc[:, 'year'] = year
    df_unemp.loc[:, 'month'] = month.values

    df_unemp.dropna(inplace=True)

    df_unemp.year = df_unemp.year.astype(int).astype(str)

    date_str = df_unemp.year + '-' + df_unemp.month
    df_unemp.loc[:, 'date'] = pd.to_datetime(date_str)

    df_unemp = df_unemp[['date', 'total', 'male', 'female']]
    df_unemp.set_index('date', inplace=True)

    # Save monthly data
    df_unemp.to_csv(save_path + 'unemp_monthly.csv')

    # Construct annual data
    group = df_unemp.groupby(df_unemp.index.year)

    # Drop incomplete years
    mask = (group.count() == 12).iloc[:, 0]
    df_unemp_annual = group.mean()[mask].round(1)
    df_unemp_annual.index = pd.to_datetime(df_unemp_annual.index.astype(str))

    # Save annual data
    df_unemp_annual.to_csv(save_path + 'unemp_annual.csv')


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    save_path = '../clean_data/'
    load_path = '../raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    unemp(params, load_path, save_path)
