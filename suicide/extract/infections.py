"""
A script to download data for the weekly cumulative count of Covid-19 infection
deaths by age and gender from the National Institute of Population and
Social Security Research (国立社会保障・人口問題研究所).

Ref: http://www.ipss.go.jp/projects/j/Choju/covid19/

"""

import os
import requests
import pandas as pd


URL = "http://www.ipss.go.jp/projects/j/Choju/covid19/data/japan_deaths.xlsx"


def infections(params, output_path):
    # Unpack parameters
    analysis_date = params['analysis_date']

    file_path = os.path.join(output_path, analysis_date, 'infections.xlsx')

    file = pd.read_excel(URL, None, header=None)
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

    for sheet_name in file.keys():
        file[sheet_name].to_excel(writer, sheet_name=sheet_name, header=None,
                                  index=None)

    writer.save()


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    output_path = "../raw_data/"

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    infections(params, output_path)
