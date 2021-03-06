"""
A script to generate output files.

"""

import os
import process
import yaml


def create_directories(directory):
    os.makedirs(os.path.join(directory, 'data_visualization'))

    forecast_types = ('aggregate', 'group')
    data_types = ('total', 'male', 'female')
    groups = ('0_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79',
              '80_99', 'total')
    dates_start = ('2009-01',
                   '2010-01',
                   '2011-01',
                   '2012-01')
    fig_types = ('present', 'full')

    dirs = [os.path.join(directory, 'model', forecast_type, data_type, group,
                         date_start, fig_type)
            for forecast_type in forecast_types
            for data_type in data_types
            for group in groups
            for fig_type in fig_types
            for date_start in dates_start]

    for d in dirs:
        os.makedirs(d)

    dirs = [os.path.join(directory, 'result_analysis', forecast_type,
                         data_type, date_start)
            for forecast_type in forecast_types
            for data_type in data_types
            for date_start in dates_start]

    for d in dirs:
        os.makedirs(d)

    os.makedirs(os.path.join(directory, 'result_analysis', 'regression_analysis'))


if __name__ == '__main__':
    params_path = 'parameters.yml'
    output_path = 'output'
    clean_data_path = 'clean_data'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    analysis_date = params['analysis_date']
    print("Start generating figures for " + analysis_date + '.')

    # Load data
    dfs = process.load_data(params, clean_data_path)
    print("Load data: done.")

    # Create directories if they don't exist
    directory = os.path.join(output_path, analysis_date)
    if not os.path.exists(directory):
        print("Create directory for " + analysis_date + ": done.")
        create_directories(directory)

    # Generate figures
    # process.visualize_data(dfs, params, output_path)
    # print("Generate data visualization figures: done.")

    process.run_model(dfs, params, output_path)
    print("Generate model figures: done.")

    process.analyze_results(params, output_path, clean_data_path)
    print("Generate regression analysis: done.")
