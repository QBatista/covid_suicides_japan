
import audit
import yaml


# TODO(QBatista): PEP8

if __name__ == '__main__':
    params_path = 'parameters.yml'
    clean_data_path = 'clean_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    print("Start auditing data for " + params['analysis_date'] + '.')

    audit.test_forecast(params, clean_data_path)
    print("Audit forecast data: done.")

    audit.test_unemp_annual(params, clean_data_path)
    print("Audit annual unemployment data: done.")

    audit.test_unemp_monthly(params, clean_data_path)
    print("Audit monthly unemployment data: done.")

    audit.test_suicide_annual(params, clean_data_path)
    print("Audit annual suicide data: done.")

    audit.test_suicide_monthly(params, clean_data_path)
    print("Audit monthly suicide data: done.")
