"""
A script to audit the forecast data.

"""

import petl as etl

analysis_date = '210803'

header = ('date', 'pre_covid', 'post_covid')

constraints = [{'name': 'forecast_date_valid', 'field': 'date',
                'test': etl.dateparser('%Y-%m-%d')},
               {'name': 'forecast_pre_covid_nonneg',
                'field': 'pre_covid', 'assertion': lambda x: float(x) >= 0},
               {'name': 'forecast_post_covid_nonneg',
                'field': 'post_covid',
                'assertion': lambda x: x == '' or float(x) >= 0}]

path = '../../clean_data/' + analysis_date + '/forecast.csv'
table = etl.csv.fromcsv(path)

problems = etl.validate(table, constraints=constraints, header=header)

if len(problems) > 1:
    print(problems.lookall())
