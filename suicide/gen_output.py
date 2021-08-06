
import process
import yaml


if __name__ == '__main__':
    params_path = 'parameters.yml'
    output_path = 'output/'
    clean_data_path = 'clean_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    print("Start generating figures for " + params['analysis_date'] + '.')

    dfs = process.load_data(params, clean_data_path)
    print("Load data: done.")

    process.fig_1(dfs, params, output_path)
    print("Generate figure 1: done.")

    process.fig_2(dfs, params, output_path)
    print("Generate figure 2: done.")

    process.fig_3_to_8(dfs, params, output_path)
    print("Generate figures 3 to 8: done.")

    process.fig_model(dfs, params, output_path)
    print("Generate model figures: done.")
