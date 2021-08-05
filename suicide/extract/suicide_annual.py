"""
A script to download data for the total, male, and female number of suicides
at an annual frequency from e-Stat.

"""

import requests
import os

# Parameters
analysis_date = '210803'

# Create directory if it doesn't exist
file_path = "../raw_data/" + analysis_date + '/'

try:
    os.stat(file_path)
except:
    os.mkdir(file_path)

csv_url = "https://www.e-stat.go.jp/en/stat-search/file-download?statInfId=000031982751&fileKind=1"

req = requests.get(csv_url)
url_content = req.content
csv_file = open(file_path + 'suicide_annual.csv', 'wb')

csv_file.write(url_content)
csv_file.close()
