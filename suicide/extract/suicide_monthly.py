"""
A script to scrape data for the total, male, and female number of suicides
at a monthly frequency from the Metropolitan Police Department (警視庁).

Ref: https://www.npa.go.jp/publications/statistics/safetylife/jisatsu.html

"""

import pandas as pd
import tabula

from bs4 import BeautifulSoup  # for scraping
import urllib.request as req
import urllib
import os
from urllib.parse import urljoin


# Parameters
analysis_date = '210803'
last_link_date = '202106'

# Create directory if it doesn't exist
file_path = "../raw_data/" + analysis_date + '/'

try:
    os.stat(file_path)
except:
    os.mkdir(file_path)

# path setting
URL = 'https://www.npa.go.jp/publications/statistics/safetylife/jisatsu.html'

# select "a" tag
res = req.urlopen(URL)
soup = BeautifulSoup(res, "html.parser")
result = soup.select("a[href]")

# create a list of links
link_list = []
for link in result:
    href = link.get("href")
    link_list.append(href)

# Take out only pdf files
pdf_list = [temp for temp in link_list if temp.endswith('pdf')]

# select the pdf file
dbpdf_list = ['/safetylife/seianki/jisatsu/R03/' + last_link_date +
              'sokuhouti.pdf',
              '/safetylife/seianki/jisatsu/R03/R02_jisatuno_joukyou.pdf',
              '/safetylife/seianki/jisatsu/R02/R01_jisatuno_joukyou.pdf',
              '/safetylife/seianki/jisatsu/H30/H30_jisatunojoukyou.pdf',
              '/safetylife/seianki/jisatsu/H29/H29_jisatsunojoukyou_01.pdf',
              '/safetylife/seianki/jisatsu/H28/H28_jisatunojoukyou_01.pdf',
              '/safetylife/seianki/jisatsu/H27/H27_jisatunojoukyou_01.pdf',
              '/safetylife/seianki/jisatsu/H26/H26_jisatunojoukyou_01.pdf',
              '/safetylife/seianki/jisatsu/H25/H25_jisatunojoukyou_01.pdf',
              '/safetylife/seianki/jisatsu/H24/H24_jisatunojoukyou_02.pdf',
              '/safetylife/seianki/jisatsu/H23/H23_jisatunojoukyou_02.pdf',
              '/safetylife/seianki/jisatsu/H22/H22_jisatunogaiyou.pdf',
              '/safetylife/seianki/jisatsu/H21/H21_jisatunogaiyou.pdf',
              '/safetylife/seianki/jisatsu/H20/H20_jisatunogaiyou.pdf']

# select the page number that has the monthly data for the number of suicides
# (total/male/female)
page_num_list = ['1', '33', '33', '34', '35', '35', '15', '15', '15', '11',
                 '11', '8', '8', '8']

# URL + pdf file name
abs_dbpdf_list = []
for relative in dbpdf_list:
    temp_url = urljoin(URL, relative)
    abs_dbpdf_list.append(temp_url)

# File name
filename_list = []
for target in abs_dbpdf_list:
    temp_list = target.split("/")
    filename_list.append(temp_list[len(temp_list)-1].replace('.pdf', '.csv'))

year_list = list(map(str, range(2021, 2007, -1)))
df_all = pd.DataFrame()

for i_pdf in range(len(abs_dbpdf_list)):
    dfs = tabula.read_pdf(abs_dbpdf_list[i_pdf], pages=page_num_list[i_pdf])

    df = dfs[0]
    df = df.drop(['Unnamed: 0', 'Unnamed: 1', '合 計'], axis=1)
    df = df.rename(columns={'1月': ''.join([year_list[i_pdf], '-01']),
                            '2月': ''.join([year_list[i_pdf], '-02']),
                            '3月': ''.join([year_list[i_pdf], '-03']),
                            '4月': ''.join([year_list[i_pdf], '-04']),
                            '5月': ''.join([year_list[i_pdf], '-05']),
                            '6月': ''.join([year_list[i_pdf], '-06']),
                            '7月': ''.join([year_list[i_pdf], '-07']),
                            '8月': ''.join([year_list[i_pdf], '-08']),
                            '9月': ''.join([year_list[i_pdf], '-09']),
                            '10月': ''.join([year_list[i_pdf], '-10']),
                            '11月': ''.join([year_list[i_pdf], '-11']),
                            '12月': ''.join([year_list[i_pdf], '-12'])})

    # Append the imported data to the front
    df_all = pd.concat([df, df_all], axis=1)

# Save data
df_all.to_csv(file_path + 'suicide_monthly.csv')
