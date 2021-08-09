"""
A script to download data for the total, male, and female number of suicides
at an annual frequency from e-Stat.

"""

import requests


URL = "https://www.e-stat.go.jp/en/stat-search/file-download?statInfId=000031982751&fileKind=1"


def suicide_annual(params, output_path):
    # Unpack parameters
    analysis_date = params['analysis_date']

    file_path = output_path + analysis_date + '/'

    req = requests.get(URL)
    url_content = req.content
    csv_file = open(file_path + 'suicide_annual.csv', 'wb')

    csv_file.write(url_content)
    csv_file.close()


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    output_path = "../raw_data/"

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_annual(params, output_path)
