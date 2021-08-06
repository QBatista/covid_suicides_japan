
import transform
import yaml


# TODO(QBatista): PEP8

if __name__ == '__main__':
    params_path = 'parameters.yml'
    raw_data_path = 'raw_data/'
    clean_data_path = 'clean_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    print("Start transforming data for " + params['analysis_date'] + '.')

    directory = clean_data_path + analysis_date + '/'
    if not os.path.exists(directory):
        print("Create directory for " + analysis_date + ": done.")
        os.makedirs(directory)

    transform.unemp(params, raw_data_path, clean_data_path)
    print("Clean unemployment data: done.")

    transform.suicide_annual(params, raw_data_path, clean_data_path)
    print("Clean annual suicide data: done.")

    transform.suicide_monthly(params, raw_data_path, clean_data_path)
    print("Clean monthly suicide data: done.")
