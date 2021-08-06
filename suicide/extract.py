
import extract
import yaml


# TODO(QBatista): Handle warnings

if __name__ == '__main__':
    params_path = 'parameters.yml'
    raw_data_path = 'raw_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    print("Start extracting data for " + params['analysis_date'] + '.')

    extract.unemp(params, raw_data_path)
    print("Extract unemployment data: done.")

    extract.suicide_annual(params, raw_data_path)
    print("Extract annual suicide data: done.")

    extract.suicide_monthly(params, raw_data_path)
    print("Extract monthly suicide data: done.")

    extract.forecast(params, raw_data_path)
    print("Extract forecast data: done.")
