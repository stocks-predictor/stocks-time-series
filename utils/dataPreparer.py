# -*- encoding: utf-8 -*-
import json, csv, os, math
import datetime as dt
from PlotSeries import plotSeriePositivesNegatives
import urllib.request
import pandas as pd
from sklearn.preprocessing import minmax_scale
import numpy as np
import argparse

# Pega o json com notícias classificadas, extrai dele uma série temporal de taxa de notícias positivas
# Pega o .csv do histórico de valores da bolsa e cria uma série temporal dele
# E gera um arquivo de saída com as duas séries.
parser = argparse.ArgumentParser(description='time series stocks predictor')
parser.add_argument('-cf', '--classifiedFile', type=str, default= '/home/matheusgomes/TCC/stocks-time-serie/utils/datasetEconomyClassified.json', help='classified file path')
parser.add_argument('-tk', '--ticker', type=str, default= 'DJI', help='ticker to get time serie')
parser.add_argument('-it', '--interval', type=str, default= '60min', help='ticker to get time serie EX: daily, 60min (default), 120min')
parser.add_argument('-o', '--out', type=str, default= '/home/matheusgomes/TCC/stocks-time-serie/utils/dataSerieS&P500.json', help='file of time serie')

IN_FILE = parser.parse_args().classifiedFile
OUT_X_Y_SERIE = parser.parse_args().out

def jsonToSerie(intervalMinutes):
    intervalSeg = intervalMinutes * 60

    if not os.path.exists(IN_FILE):
        unionPredictionsWithData()
    
    jsonData = []
    
    with open(IN_FILE) as jFile:
        jsonData = json.load(jFile)
    
    seriePoints = []

    currentDateTime = ''
    refDateTime = math.floor(float(jsonData[0]['timestamp'])/intervalSeg)
    pos = 0
    neg = 0
    for data in jsonData:
        currentDateTime = math.floor(float(data['timestamp'])/intervalSeg)
        if (currentDateTime == refDateTime):
            if (data['classification'] == 1):
                pos = pos + 1
            else:
                neg = neg + 1     
        else:
            tot = pos + neg
            print ('currentDateTime: %s  pos(tx): %.2f, neg(tx): %.2f'%(dt.datetime.utcfromtimestamp(float(data['timestamp'])).strftime('%Y-%m-%d %H:%M'), float(pos)/tot, float(neg)/tot))
            seriePoints.append({'timestamp': data['timestamp'], 'pos(qtd)': pos, 'pos(rate)': float(pos)/tot, 'neg(qtd)': neg, 'neg(rate)': float(neg)/tot})
            pos = 0
            neg = 0
            refDateTime = math.floor(float(data['timestamp'])/intervalSeg)
            if (data['classification'] == 1):
                pos = pos + 1
            else:
                neg = neg + 1

    with open(OUT_X_Y_SERIE, 'w') as f:
        f.write(json.dumps(seriePoints, indent = 4))

def getStocksExchangeIndex(ticker = 'DJI', interval = '60min'):
    api_key = 'LGIJWKLPWQMN0ZO0'
    
    if (interval == 'daily'):
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&apikey=%s"%(ticker, api_key)

        # Save data to this file
        file_to_save = 'stock_market_data-%s-daily.csv'%ticker
        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date', 'Timestamp','Low','High','Close','Open'])
                for k,v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date, date.timestamp(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1

            print('Data saved to : %s'%file_to_save)        
            df.to_csv(file_to_save)
        
        else:
            print('File already exists. Loading data from CSV')
            df = pd.read_csv(file_to_save)    
    else:
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=%s&apikey=%s"%(ticker, interval, api_key)

        # Save data to this file
        file_to_save = 'stock_market_data-%s.csv'%ticker
        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (%s)'%(interval)]
                df = pd.DataFrame(columns=['Date', 'Timestamp','Low','High','Close','Open'])
                for k,v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d %H:%M:%S')
                    data_row = [date, date.timestamp(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1

            print('Data saved to : %s'%file_to_save)        
            df.to_csv(file_to_save)
        
        else:
            print('File already exists. Loading data from CSV')
            df = pd.read_csv(file_to_save)

    # Sort DataFrame by date
    df = df.sort_values('Date')

    timestamps = df.loc[:, 'Timestamp'].as_matrix()
    high_prices = df.loc[:,'High'].as_matrix()
    low_prices = df.loc[:,'Low'].as_matrix()
    mid_prices = (high_prices+low_prices)/2.0
    mid_prices_norm = minmax_scale(mid_prices) # normalizado entre [0, 1]

    return timestamps, mid_prices_norm

def unionDataSeriePosNeg_stocksExchangeSerie(timestampStocks, midPricesStocks, dataSeriePosNeg, intervalMinutes):
    def search(timestampStocks, timeFloor, intervalSeg):
        idx = 0
        for t in timestampStocks:
            if (timeFloor == math.floor(float(t) / intervalSeg)):
                return idx
            idx = idx + 1
        return -1

    intervalSeg = 60 * intervalMinutes

    dataPosNeg = json.load(open(dataSeriePosNeg, "r"))
    
    x, y = [], []
    
    midPriceAnterior = 0
    i = 0
    for d in dataPosNeg:
        timeFloor = math.floor(float(d['timestamp']) / intervalSeg)
        
        r = search(timestampStocks, timeFloor, intervalSeg)
        if (r == -1):
            d['midPriceStocks'] = midPriceAnterior
        else:
            d['midPriceStocks'] = midPricesStocks[r]
            midPriceAnterior = midPricesStocks[r]

    with open(dataSeriePosNeg, 'w') as f:
        f.write(json.dumps(dataPosNeg, indent = 4))
    
if __name__ == '__main__':

    if (parser.parse_args().interval == 'daily'):
        intervalMinutes = 1440
    else:
        n = int((parser.parse_args().interval).split('m')[0])
        intervalMinutes = n

    jsonToSerie(intervalMinutes = intervalMinutes)

    timestampStocks, midPricesStocks = getStocksExchangeIndex(ticker=parser.parse_args().ticker, interval=parser.parse_args().interval)
    unionDataSeriePosNeg_stocksExchangeSerie(timestampStocks.tolist(), midPricesStocks.tolist(), OUT_X_Y_SERIE, intervalMinutes)

    plotSeriePositivesNegatives(OUT_X_Y_SERIE)