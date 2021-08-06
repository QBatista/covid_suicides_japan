"""
A script to audit the monthly suicide data.

"""

import petl as etl


def test_suicide_monthly(params, load_path):
    analysis_date = params['analysis_date']

    header = ('date', 'total', 'male', 'female')

    def valid_assert_map(x): return float(x[0]) == float(x[1]) + float(x[2])

    constraints = [{'name': 'suicide_monthly_date_valid', 'field': 'date',
                    'test': etl.dateparser('%Y-%m-%d')},
                   {'name': 'suicide_monthly_total_float',
                    'field': 'total', 'test': float},
                   {'name': 'suicide_monthly_male_float',
                    'field': 'male', 'test': float},
                   {'name': 'suicide_monthly_female_float',
                    'field': 'female', 'test': float},
                   {'name': 'suicide_monthly_total_equals_male_plus_female',
                    'field': ['total', 'male', 'female'],
                    'assertion': valid_assert_map},
                   {'name': 'suicide_monthly_not_none',
                    'assertion': lambda row: None not in row}]

    path = load_path + analysis_date + '/suicide_monthly.csv'
    table = etl.csv.fromcsv(path)

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

    test_suicide_monthly(params, load_path)
