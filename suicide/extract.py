"""
A script for extracting raw data.

"""

import os
import yaml
import extract


# TODO(QBatista): Fix warnings

if __name__ == '__main__':
    params_path = 'parameters.yml'
    raw_data_path = 'raw_data'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    analysis_date = params['analysis_date']
    print("Start extracting data for " + analysis_date + '.')

    directory = os.path.join(raw_data_path, analysis_date)
    if not os.path.exists(directory):
        print("Create directory for " + analysis_date + ": done.")
        os.makedirs(directory)

    extract.unemp_dist(params, raw_data_path)
    print("Extract unemployment distribution data: done.")

    suicide_dist_directory = os.path.join(directory, 'suicide_dist')
    if not os.path.exists(suicide_dist_directory):
        annual_dir = os.path.join(suicide_dist_directory, 'annual')
        os.makedirs(annual_dir)

        monthly_dir = os.path.join(suicide_dist_directory, 'monthly')
        os.makedirs(monthly_dir)

    extract.suicide_annual(params, raw_data_path)
    print("Extract annual suicide data: done.")

    extract.suicide_monthly(params, raw_data_path)
    print("Extract monthly suicide data: done.")

    forecast_directory = os.path.join(directory, 'forecast')
    if not os.path.exists(forecast_directory):
        os.makedirs(forecast_directory)

    extract.forecast(params, raw_data_path)
    print("Extract forecast data: done.")

    extract.life_expectancy(params, raw_data_path)
    print("Extract life expectancy data: done.")

    extract.infections(params, raw_data_path)
    print("Extract infection deaths data: done.")

    print("Data Extraction Successfully Completed!")
