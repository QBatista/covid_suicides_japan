"""
A script to download data for the unemployment rate for different genders and
age groups from the Statistics Bureau of Japan (総務省統計局).

Ref: https://www.stat.go.jp/data/roudou/longtime/03roudou.html#hyo_1

"""

import os
import requests


URL = 'https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a90.xlsx'


def unemp_dist(params, output_path):
    """
    Download unemployment rate distribution data and save it to the
    `output_path` folder based on `params['analysis_date']`.

    """

    # Unpack parameters
    analysis_date = params['analysis_date']

    # Get file
    req = requests.get(URL)
    url_content = req.content

    # Save file
    file_path = os.path.join(output_path, analysis_date, 'unemp_dist.xlsx')
    file = open(file_path, 'wb')
    file.write(url_content)
    file.close()


if __name__ == '__main__':
    import yaml

    params_path = os.path.join(os.pardir, 'parameters.yml')
    output_path = os.path.join(os.pardir, 'raw_data')

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    unemp_dist(params, output_path)
