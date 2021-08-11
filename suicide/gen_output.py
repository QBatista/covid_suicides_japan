"""
A script to generate output files.

"""

import os
import process
import yaml


def create_directories(directory):
    os.makedirs(directory + 'predictions/')
    os.makedirs(directory + 'figures/annual/')

    forecast_types = ('agg_forecast', 'group_forecast')
    data_types = ('total', 'male', 'female')
    age_groups = ('0_19', '20_29', '30_39', '40_49', '50_59', '60_69',
                  '70_79', '80_99', 'total')
    fig_types = ('present', 'full')
    dates_start = ('2009-01',
                   '2010-01',
                   '2011-01',
                   '2012-01')

    dirs = [directory + 'figures/' + forecast_type + '/' + data_type + '/' +
            age_group + '/' + fig_type + '/' + date_start + '/'
            for forecast_type in forecast_types
            for data_type in data_types
            for age_group in age_groups
            for fig_type in fig_types
            for date_start in dates_start]

    for d in dirs:
        os.makedirs(d)


if __name__ == '__main__':
    params_path = 'parameters.yml'
    output_path = 'output/'
    clean_data_path = 'clean_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    analysis_date = params['analysis_date']
    print("Start generating figures for " + analysis_date + '.')

    dfs = process.load_data(params, clean_data_path)
    print("Load data: done.")

    # Create directories if they don't exist
    directory = output_path + analysis_date + '/'
    if not os.path.exists(directory):
        print("Create directory for " + analysis_date + ": done.")
        create_directories(directory)

    # Generate figures
    process.fig_1(dfs, params, output_path)
    print("Generate figure 1: done.")

    process.fig_2(dfs, params, output_path)
    print("Generate figure 2: done.")

    process.fig_3_to_8(dfs, params, output_path)
    print("Generate figures 3 to 8: done.")

    process.run_model(dfs, params, output_path)
    print("Generate model figures: done.")
