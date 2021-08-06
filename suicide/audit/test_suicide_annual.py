"""
A script to audit the annual suicide data.

"""

import petl as etl


# TODO(QBatista): PEP8

def test_suicide_annual(params, load_path):
    analysis_date = params['analysis_date']

    header = ('date', 'total', 'male', 'female')

    valid_assert_map = lambda x: float(x[0]) == float(x[1]) + float(x[2])

    constraints = [{'name': 'suicide_annual_date_valid', 'field': 'date',
                    'test': etl.dateparser('%Y-%m-%d')},
                   {'name': 'suicide_annual_total_float',
                    'field': 'total', 'test': float},
                   {'name': 'suicide_annual_male_float',
                    'field': 'male', 'test': float},
                   {'name': 'suicide_annual_female_float',
                    'field': 'female', 'test': float},
                   {'name': 'suicide_annual_total_equals_male_plus_female',
                    'field': ['total', 'male', 'female'],
                    'assertion': valid_assert_map},
                   {'name': 'suicide_annual_not_none',
                    'assertion': lambda row: None not in row}]

    path = load_path + analysis_date + '/suicide_annual.csv'
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

    test_suicide_annual(params, load_path)
