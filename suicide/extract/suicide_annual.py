"""
A script to download data annual suicide data at an annual frequency from
the Ministry of Health, Labor, and Welfare (厚生労働省).

Ref: https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000140901.html

"""

import os
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile


URL = "https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000140901.html"
HOME = "https://www.mhlw.go.jp"


def get_csvs2(url, year, save_path):
  html=requests.get(url)
  soup=BeautifulSoup(html.content,'html.parser')
  for link in soup.find_all("a"):
    if "確定値その２" in link.text:
      r = requests.get(HOME+link.get("href"), stream=True)

      path = os.path.join(save_path, str(year)+".zip")
      with open(path, 'wb') as fd:
          for chunk in r.iter_content(chunk_size=128):
              fd.write(chunk)


def suicide_annual(params, output_path):
    # Unpack parameters
    analysis_date = params['analysis_date']

    save_path = os.path.join(output_path, analysis_date, 'suicide_dist', 'annual')

    html = requests.get(URL)
    soup = BeautifulSoup(html.content, 'html.parser')

    year=2009
    for link in soup.find(class_="m-listLink").find_all("a"):
        if "（平成" in link.text or "（令和２年" in link.text:
            print(link.text)
            get_csvs2(HOME+link.get("href"),year, save_path)
            year+=1

    for year in range(2009,2021):
        filename = str(year)+".zip"
        path = os.path.join(save_path, filename)
        with ZipFile(path, 'r') as zipObj:
            for file in zipObj.namelist():
                if "B1-4" in file:
                    zipObj.extract(file, save_path)
                    member = os.path.join(save_path, file)
                    extract_path = os.path.join(save_path, str(year)+".xls")
                    os.rename(member,extract_path)
                    break
        os.remove(path)


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    output_path = "../raw_data/"

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_annual(params, output_path)
