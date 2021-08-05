"""
A script to clean the unemployment data.

"""

import pandas as pd
import os


# Parameters
analysis_date = '210803'

# Create directory if it doesn't exist
file_path = "../clean_data/" + analysis_date + '/'

# Load data
load_path = '../raw_data/' + analysis_date + '/'
df = pd.read_excel(load_path + 'unemp.xlxs', skiprows=8)

df_unemp = df[['Both sexes.5', 'Male.5', 'Female.5']].copy()
df_unemp.columns = ['total', 'male', 'female']

year = pd.to_numeric(df['Unnamed: 0'].shift(-1),
                     errors='coerce').fillna(method='ffill')
month = df['Unnamed: 1'].str.extract('(\d+)')

df_unemp.loc[:, 'year'] = year
df_unemp.loc[:, 'month'] = month.values

df_unemp.dropna(inplace=True)

df_unemp.year = df_unemp.year.astype(int).astype(str)

df_unemp.loc[:, 'date'] = pd.to_datetime(df_unemp.year + '-' + df_unemp.month)

df_unemp = df_unemp[['date', 'total', 'male', 'female']]
df_unemp.set_index('date', inplace=True)

df_unemp.to_csv(file_path + 'unemp_monthly.csv')

group = df_unemp.groupby(df_unemp.index.year)
mask = (group.count() == 12).iloc[:, 0]
df_unemp_annual = group.mean()[mask]
df_unemp_annual.index = pd.to_datetime(df_unemp_annual.index.astype(str))

# Save data
df_unemp_annual.to_csv(file_path + 'unemp_annual.csv')
