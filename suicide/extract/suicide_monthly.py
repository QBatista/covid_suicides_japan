"""
A script to scrape data for the number of suicides for different genders and
age groups at a monthly frequency from Ministry of Health, Labor, and
Welfare (厚生労働省).

Ref: https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000140901.html

"""

import os
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile


URL_HOME = "https://www.mhlw.go.jp"
URL_DATA = "https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000140901.html"
START_YEAR = 2009  # Initial year for the data


def get_zip(url, year, save_path):
    """
    Download monthly suicide data as zip files for a given `year` and save
    them to `save_path`.

    Parameters
    ----------
    url : str
        URL for the MHLW webpage with monthly suicide data for `year`.

    year : int
        Year for the monthly suicide data.

    save_path : str
        Path for saving the zip file.

    """

    # Get html
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')

    month = 1
    for link in soup.find_all("a"):
        if "（暫定値）" in link.text or "（確定値）" in link.text:
            month_url = URL_HOME + link.get("href")
            r = requests.get(month_url, stream=True)

            # Save zip file
            name = str(year) + "-" + f"{month:02d}" + ".zip"
            full_path = os.path.join(save_path, name)
            with open(full_path, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=128):
                    fd.write(chunk)

            # Increment month count
            month += 1


def extract_zip(year, save_path):
    """
    Extract all existing zip files for monthly suicide data in `year`, save
    the output to `save_path`, and delete extracted zip files.

    """

    for month in range(1, 13):
        filename = str(year) + "-" + f"{month:02d}"
        path = os.path.join(save_path, filename + ".zip")

        # Check if zip file exists
        if os.path.exists(path):
            # Extract zip file
            with ZipFile(path, 'r') as zipObj:
                for file in zipObj.namelist():
                    if "A1-4" in file:
                        zipObj.extract(file, save_path)
                        member = os.path.join(save_path, file)
                        full_filename = filename + ".xls"
                        extract_path = os.path.join(save_path, full_filename)
                        os.rename(member, extract_path)
                        break
            # Delete zip file
            os.remove(path)

def suicide_monthly(params, output_path):
    """
    Download monthly suicide data and save it to the `output_path` folder
    based on `params['analysis_date']`.

    """

    # Unpack parameters
    analysis_date = params['analysis_date']

    save_path = os.path.join(output_path, analysis_date, 'suicide_dist', 'monthly')

    html = requests.get(URL_DATA)
    soup = BeautifulSoup(html.content, 'html.parser')

    # Set initial year
    year = START_YEAR
    for link in soup.find(class_="m-listLink").find_all("a"):
        if "（平成" in link.text or "（令和" in link.text:
            year_url = URL_HOME + link.get("href")
            get_zip(year_url, year, save_path)

            # Increment year count
            year += 1

    # Extract all zip files
    for y in range(START_YEAR, 2021 + 1):
        extract_zip(y, save_path)


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    output_path = "../raw_data/"

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    suicide_monthly(params, output_path)
