
import process
import yaml

if __name__ == '__main__':
    params_path = 'parameters.yml'
    output_path = 'output/'
    clean_data_path = 'clean_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    dfs = process.load_data(params, clean_data_path)
    process.fig_1(dfs, params, output_path)
    process.fig_2(dfs, params, output_path)
