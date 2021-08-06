"""
A script to audit the forecast data.

"""

import petl as etl


# TODO(QBatista): PEP8

def test_forecast(params, load_path):
    analysis_date = params['analysis_date']
    load_path += analysis_date + '/forecast.csv'

    header = ('date', 'pre_covid', 'post_covid')

    constraints = [{'name': 'forecast_date_valid', 'field': 'date',
                    'test': etl.dateparser('%Y-%m-%d')},
                   {'name': 'forecast_pre_covid_nonneg',
                    'field': 'pre_covid', 'assertion': lambda x: float(x) >= 0},
                   {'name': 'forecast_post_covid_nonneg',
                    'field': 'post_covid',
                    'assertion': lambda x: x == '' or float(x) >= 0}]

    table = etl.csv.fromcsv(load_path)

    problems = etl.validate(table, constraints=constraints, header=header)

    if len(problems) > 1:
        print(problems.lookall())


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    load_path = '../clean_data/'

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    test_forecast(params, load_path)
