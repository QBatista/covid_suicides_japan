"""
A script to clean the raw annual suicide data.

"""

import pandas as pd


def suicide_annual(params, load_path, save_path):
    # Unpack arguments
    analysis_date = params['analysis_date']

    save_path += analysis_date + '/'
    load_path += analysis_date + '/'

    # Load raw data
    df = pd.read_csv(load_path + 'suicide_annual.csv',
                     encoding='unicode_escape',
                     skiprows=14)

    # Select year/type and suicide columns
    df_suicide = df[['Unnamed: 0', 'Deaths.16']].copy()

    # Rename columns
    df_suicide.columns = ['date', 'suicides']

    # Find types (total, male, female)
    type_data = df_suicide.date[df_suicide.suicides.isna()].str.lower()

    # Construct a columns of types
    df_suicide.loc[type_data.index, 'type'] = type_data.values
    df_suicide.type.fillna(method='ffill', inplace=True)
    df_suicide.dropna(inplace=True)

    # Pivot data
    df_suicide = df_suicide.pivot(index='date', columns='type',
                                  values='suicides')
    df_suicide = df_suicide[['total', 'male', 'female']]

    # Use datetime index
    df_suicide.index = pd.to_datetime(df_suicide.index)

    # Save data to csv
    df_suicide.to_csv(save_path + 'suicide_annual.csv')


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    save_path = '../clean_data/'
    load_path = '../raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_annual(params, load_path, save_path)
