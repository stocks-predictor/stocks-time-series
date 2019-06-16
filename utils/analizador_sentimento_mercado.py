# -*- encoding: UTF-8 -*-
import json, csv
from functools import reduce
import datetime as dt

IN_FILE = '/home/matheusgomes/TCC/stocks-time-serie/utils/dataSerieDJI_02_05_a_21_05_60min.json'

n_ant = 5

with open(IN_FILE) as sf:
    serieData = json.load(sf)

def do_sum(x1, x2): return x1 + x2

fileMercado = open('sentimento_mercado_02_05_a_21_05.csv', mode='w')
sent_mercado_writer = csv.writer(fileMercado, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
sent_mercado_writer.writerow(['média notícias positivas', 'preço médio ações (anterior)', 'preço médio ações (atual)', 'data e hora atual'])

idx = 0
mps_ant = serieData[0]['midPriceStocks']
posRate_ant = []

for p in serieData:
    if (len(posRate_ant) < n_ant):
        posRate_ant.insert(0, p['pos(rate)'])
    else:
        posRate_ant.pop()
        posRate_ant.insert(0, p['pos(rate)'])
    
    if (mps_ant != p['midPriceStocks']):
        # soma todos os valores do vetor
        media_posRate = reduce(do_sum, posRate_ant) / len(posRate_ant)
        
        print('* media positivos: %f,  preço médio anterior: %f, preço médio atual: %f ...  data/hora Atual: %s'%(media_posRate, mps_ant, p['midPriceStocks'], dt.datetime.utcfromtimestamp(float(p['timestamp'])).strftime('%Y-%m-%d %H:%M')))
        sent_mercado_writer.writerow([media_posRate, mps_ant, p['midPriceStocks'], dt.datetime.utcfromtimestamp(float(p['timestamp'])).strftime('%Y-%m-%d %H:%M')])
        
        mps_ant = p['midPriceStocks']
    
    idx = idx + 1
