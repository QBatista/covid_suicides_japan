"""
A script to audit the annual unemployment data.

"""

import petl as etl

analysis_date = '210803'

header = ('date', 'total', 'male', 'female')

constraints = [{'name': 'unemp_annual_date_valid', 'field': 'date',
                'test': etl.dateparser('%Y-%m-%d')},
               {'name': 'unemp_annual_total_float',
                'field': 'total', 'test': float},
               {'name': 'unemp_annual_male_float',
                'field': 'male', 'test': float},
               {'name': 'unemp_annual_female_float',
                'field': 'female', 'test': float},
               {'name': 'unemp_annual_not_none',
                'assertion': lambda row: None not in row},
               {'name': 'unemp_annual_total_nonneg',
                'field': 'total',
                'assertion': lambda x: float(x) >= 0},
               {'name': 'unemp_annual_male_nonneg',
                'field': 'male',
                'assertion': lambda x: float(x) >= 0},
               {'name': 'unemp_annual_female_nonneg',
                'field': 'female',
                'assertion': lambda x: float(x) >= 0}]

path = '../../clean_data/' + analysis_date + '/unemp_annual.csv'
table = etl.csv.fromcsv(path)

problems = etl.validate(table, constraints=constraints, header=header)

if len(problems) > 1:
    print(problems.lookall())
