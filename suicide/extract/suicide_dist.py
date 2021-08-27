"""
A script to scrape data for the number of suicides for different genders and
age groups at a monthly frequency from Ministry of Health, Labor, and
Welfare (厚生労働省).

Ref: https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000140901.html

"""

import os
import requests
from bs4 import BeautifulSoup


URL = "https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000140901.html"
HOME = "https://www.mhlw.go.jp"


def get_csvs(url, year, save_path):
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    month = 1
    for link in soup.find_all("a"):
        if "（暫定値）" in link.text:
            r = requests.get(HOME+link.get("href"), stream=True)

            path = os.path.join(save_path, str(year)+"-"+f"{month:02d}"+".zip")
            with open(path, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=128):
                    fd.write(chunk)
            month += 1


def suicide_dist(params, output_path):
    analysis_date = params['analysis_date']

    save_path = os.path.join(output_path, analysis_date, 'suicides_dist')

    html = requests.get(URL)
    soup = BeautifulSoup(html.content, 'html.parser')

    year=2009
    for link in soup.find(class_="m-listLink").find_all("a"):
        if "（平成" in link.text or "（令和" in link.text:
            print(link.text)
            get_csvs(HOME+link.get("href"),year, save_path)
            year += 1



if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    output_path = "../raw_data/"

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_dist(params, output_path)
