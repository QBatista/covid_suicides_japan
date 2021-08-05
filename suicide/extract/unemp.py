"""
A script to download data for the total, male, and female unemployment rate
at a monthly frequency the Statistics Bureau of Japan　(総務省統計局).

Ref: https://www.stat.go.jp/data/roudou/longtime/03roudou.html#hyo_1

"""

import requests


# Parameters
analysis_date = '210803'

# Create directory if it doesn't exist
file_path = "../raw_data/" + analysis_date + '/'

csv_url = 'https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a10.xlsx'

req = requests.get(csv_url)
url_content = req.content
csv_file = open(file_path + 'unemp.xlxs', 'wb')

csv_file.write(url_content)
csv_file.close()
