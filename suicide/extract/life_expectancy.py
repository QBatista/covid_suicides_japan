"""
A script to download data for life expectancy by age for different genders
from the Ministry of Health, Labor, and Welfare (厚生労働省).

Ref: https://www.mhlw.go.jp/toukei/saikin/hw/life/life19/index.html

"""

import os
import requests
import pandas as pd


URL = 'https://www.mhlw.go.jp/toukei/saikin/hw/life/life19/xlsx/life19-12.xlsx'


def life_expectancy(params, output_path):
    """
    Download life expectancy data and save it to the `output_path` folder
    and `params['analysis_date']`.

    """

    # Unpack parameters
    analysis_date = params['analysis_date']

    # Get file
    file = pd.read_excel(URL, None, header=None)

    # Save file
    fpath = os.path.join(output_path, analysis_date, 'life_expectancy.xlsx')
    writer = pd.ExcelWriter(fpath, engine='xlsxwriter')
    for sheet_name in file.keys():
        file[sheet_name].to_excel(writer, sheet_name=sheet_name, header=None,
                                  index=None)
    writer.save()


if __name__ == '__main__':
    import yaml

    params_path = os.path.join(os.pardir, 'parameters.yml')
    output_path = os.path.join(os.pardir, 'raw_data')

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    life_expectancy(params, output_path)
