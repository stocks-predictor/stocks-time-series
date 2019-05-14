# -*- encoding: UTF-8 -*-
import pandas as pd
from pandas import Series
import datetime as dt

csvPath = '/home/matheusgomes/Downloads/DJI_2008_2016.csv'
file_to_save = 'stock_market_data-DJI-daily.csv'

df = pd.read_csv(csvPath)

df = df.sort_values('Date')

dates = df.loc[:, 'Date'].values

timestampList = []
for date in dates:
    timestamp = dt.datetime.strptime(date, '%Y-%m-%d').timestamp()
    timestampList.append(str(timestamp))

df['Timestamp']= timestampList

print(df.head())

df.to_csv(file_to_save)