# -*- encoding: utf-8 -*-
import json, csv, os, math
import datetime as dt
from PlotSeries import plotSeriePositivesNegatives
import urllib.request
import pandas as pd
from sklearn.preprocessing import minmax_scale
import numpy as np
#Pega o csv com as probabilidades preditas (PREDICTION_PATH) 
# do json de notícias a serem preditas (NEWS_PATH)
# e gera um json com a classificação das notícias (OUT_FILE)
# após isso gera um json com a série temporal com as taxas de notícias positivas e negativas (OUT_X_Y_SERIE)

PREDICTIONS_PATH = '/home/matheusgomes/TCC/stocks-time-serie/utils/predict_results.tsv'
NEWS_PATH = '/home/matheusgomes/TCC/stocks-time-serie/utils/datasetEconomyNews_to_pred.json'
OUT_FILE = '/home/matheusgomes/TCC/stocks-time-serie/utils/datasetEconomyClassified.json'
OUT_X_Y_SERIE = '/home/matheusgomes/TCC/stocks-time-serie/utils/dataSeriePosNeg.json'

def unionPredictionsWithData():
    jsonData = []
    
    with open(NEWS_PATH) as jFile:
        jsonData = json.load(jFile)

    with open(PREDICTIONS_PATH) as tsvFile:
        tsvData = csv.reader(tsvFile, delimiter='\t')
        idx = 0
        for neg, pos in tsvData:
            if float(pos) >= 0.5:
                jsonData[idx]['classification'] = 1
            else:
                jsonData[idx]['classification'] = -1

            idx = idx + 1   

    with open(OUT_FILE, 'w') as writeJson:
        writeJson.write(json.dumps(jsonData, indent = 4))
    
def jsonToSerie(intervalMinutes):
    intervalSeg = intervalMinutes * 60

    if not os.path.exists(OUT_FILE):
        unionPredictionsWithData()
    
    jsonData = []
    
    with open(OUT_FILE) as jFile:
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
    jsonToSerie(intervalMinutes = 60)
    timestampStocks, midPricesStocks = getStocksExchangeIndex()
    unionDataSeriePosNeg_stocksExchangeSerie(timestampStocks.tolist(), midPricesStocks.tolist(), OUT_X_Y_SERIE, 60)

    plotSeriePositivesNegatives(OUT_X_Y_SERIE)