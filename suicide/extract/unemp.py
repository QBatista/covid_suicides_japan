"""
A script to download data for the total, male, and female unemployment rate
at a monthly frequency the Statistics Bureau of Japan　(総務省統計局).

Ref: https://www.stat.go.jp/data/roudou/longtime/03roudou.html#hyo_1

"""

import requests


# TODO(QBatista): PEP8

URL = 'https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a10.xlsx'

def unemp(params, output_path):
    # Unpack parameters
    analysis_date = params['analysis_date']

    file_path = output_path + analysis_date + '/'

    req = requests.get(URL)
    url_content = req.content
    csv_file = open(file_path + 'unemp.xlxs', 'wb')

    csv_file.write(url_content)
    csv_file.close()


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    output_path = "../raw_data/"

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    unemp(params, output_path)
