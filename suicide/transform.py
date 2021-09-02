"""
A script to clean raw data.

"""

import os
import transform
import yaml


if __name__ == '__main__':
    params_path = 'parameters.yml'
    raw_data_path = 'raw_data'
    clean_data_path = 'clean_data'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    analysis_date = params['analysis_date']
    print("Start transforming data for " + analysis_date + '.')

    directory = clean_data_path + analysis_date + '/'
    if not os.path.exists(directory):
        print("Create directory for " + analysis_date + ": done.")
        os.makedirs(directory)

    transform.unemp_dist(params, raw_data_path, clean_data_path)
    print("Clean unemployment distribution data: done.")

    transform.suicide_dist_annual(params, raw_data_path, clean_data_path)
    print("Clean annual suicide distribution data: done.")

    transform.suicide_dist_monthly(params, raw_data_path, clean_data_path)
    print("Clean monthly suicide distribution data: done.")

    transform.infections(params, raw_data_path, clean_data_path)
    print("Clean infection deaths data: done.")

    transform.life_expectancy(params, raw_data_path, clean_data_path)
    print("Clean life expectancy data: done.")

    print("Successfully Cleaned Data!")
