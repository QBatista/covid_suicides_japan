"""
A script to clean the raw monthly suicide data.

"""

import pandas as pd
import os


# Parameters
analysis_date = '210803'

# Create directory if it doesn't exist
file_path = "../clean_data/" + analysis_date + '/'

try:
    os.stat(file_path)
except:
    os.mkdir(file_path)

# Load data
load_path = '../raw_data/' + analysis_date + '/'
df = pd.read_csv(load_path + 'suicide_monthly.csv', index_col=0)

# Transpose data
df = df.T

# Rename the columns
df.columns = ['total', 'male', 'female']

# Drop missing values
df.dropna(inplace=True)

# Convert index to datetime with "month start" frequency
df.index = pd.to_datetime(df.index).to_period('M').to_timestamp()

# Convert data to numeric values
df = df.apply(lambda x: pd.to_numeric(x.str.replace('[, ]', '', regex=True)))

# Rename index
df.index.rename('date', inplace=True)

# Save data
df.to_csv(file_path + 'suicide_monthly.csv')
