"""
A script to clean the raw annual suicide data.

"""

import pandas as pd
import os


# Parameters
analysis_date = '210803'

# Create directory if it doesn't exist
save_path = "../clean_data/" + analysis_date + '/'

try:
    os.stat(save_path)
except:
    os.mkdir(save_path)

load_path = '../raw_data/' + analysis_date + '/'
df = pd.read_csv(load_path + 'suicide_annual.csv', encoding= 'unicode_escape',
                 skiprows=14)

# Select year/type and suicide columns
df_suicide = df[['Unnamed: 0', 'Deaths.16']].copy()

# Rename columns
df_suicide.columns = ['date', 'suicides']

# Find types
type_data = df_suicide.date[df_suicide.suicides.isna()].str.lower()

# Construct a columns of types
df_suicide.loc[type_data.index, 'type'] = type_data.values
df_suicide.type.fillna(method='ffill', inplace=True)
df_suicide.dropna(inplace=True)

# Pivot data
df_suicide = df_suicide.pivot(index='date', columns='type', values='suicides')
df_suicide = df_suicide[['total', 'male', 'female']]

# Use datetime index
df_suicide.index = pd.to_datetime(df_suicide.index)

# Save data to csv
df_suicide.to_csv(save_path + 'suicide_annual.csv')
