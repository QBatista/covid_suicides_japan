"""
A script to download data for the unemployment rate for different genders and
age groups from the Statistics Bureau of Japan (総務省統計局).

Ref: https://www.stat.go.jp/data/roudou/longtime/03roudou.html#hyo_1

"""

import requests


URL = 'https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a90.xlsx'


def unemp(params, output_path):
    # Unpack parameters
    analysis_date = params['analysis_date']

    file_path = output_path + analysis_date + '/'

    req = requests.get(URL)
    url_content = req.content
    file = open(file_path + 'unemp_dist.xlsx', 'wb')

    file.write(url_content)
    file.close()


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    output_path = "../raw_data/"

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    unemp(params, output_path)