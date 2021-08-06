"""
A script for extracting raw data.

"""

import os
import yaml
import extract


if __name__ == '__main__':
    params_path = 'parameters.yml'
    raw_data_path = 'raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    analysis_date = params['analysis_date']
    print("Start extracting data for " + analysis_date + '.')

    directory = raw_data_path + analysis_date + '/'
    if not os.path.exists(directory):
        print("Create directory for " + analysis_date + ": done.")
        os.makedirs(directory)

    extract.unemp(params, raw_data_path)
    print("Extract unemployment data: done.")

    extract.suicide_annual(params, raw_data_path)
    print("Extract annual suicide data: done.")

    extract.suicide_monthly(params, raw_data_path)
    print("Extract monthly suicide data: done.")

    forecast_directory = directory + 'forecast/'
    if not os.path.exists(forecast_directory):
        os.makedirs(forecast_directory)

    extract.forecast(params, raw_data_path)
    print("Extract forecast data: done.")
