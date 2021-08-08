"""
A script to clean the raw monthly suicide data.

"""

import pandas as pd


def suicide_monthly(params, load_path, save_path):
    # Unpack arguments
    analysis_date = params['analysis_date']

    load_path += analysis_date + '/'
    save_path += analysis_date + '/'

    # Load raw data
    df = pd.read_csv(load_path + 'suicide_monthly.csv', index_col=0)

    # Transpose data
    df = df.T

    # Rename the columns
    df.columns = ['total', 'male', 'female']

    # Drop missing values
    df.dropna(inplace=True)

    # Convert index to datetime with "month start" frequency
    df.index = pd.to_datetime(df.index).to_period('M').to_timestamp()

    # Convert data to numeric values
    df = df.apply(lambda x: pd.to_numeric(x.str.replace('[, ]', '',
                                          regex=True)))

    # Rename index
    df.index.rename('date', inplace=True)

    # Save data
    df.to_csv(save_path + 'suicide_monthly.csv')


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    save_path = '../clean_data/'
    load_path = '../raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_monthly(params, load_path, save_path)
