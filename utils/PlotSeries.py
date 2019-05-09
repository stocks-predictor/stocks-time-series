import json
import os
import datetime
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import numpy as np

source = ""

def loadJson(fname):
    rawData = json.load(open(os.path.join(source, fname), "r"))
    x, ratePos, rateNeg, mps = [], [], [], []
    for point in rawData:
        x.append(datetime.datetime.utcfromtimestamp(float(point["timestamp"])).strftime('%Y-%m-%d %H:%M'))
        ratePos.append(point["pos(rate)"])
        rateNeg.append(point["neg(rate)"])
        mps.append(point["midPriceStocks"])
    return x, ratePos, rateNeg, mps 

def generatexTicks(interval, nlocs, labels):
    locs, nlabels = [], []
    for idx in range(0, nlocs, interval):
        locs.append(idx)
        nlabels.append(labels[idx])
    return locs, nlabels

def plotSeriePositivesNegatives(filePath):
    x, ratePos, rateNeg, mps = loadJson(filePath)
    
    def moving_average(signal, period):
        buffer = [np.nan] * period
        for i in range(period,len(signal)):
            buffer.append(signal[i-period:i].mean())
        return buffer
    
    #def moving_average(a, n=3) :
    #    ret = np.cumsum(a, dtype=float)
    #    ret[n:] = ret[n:] - ret[:-n]
    #    return ret[n - 1:] / n
    
    media_movel = moving_average(signal=np.asarray(ratePos), period=10)
    
    plt.figure(figsize=(10, 6))
    l1, = plt.plot(x, ratePos, color='burlywood', label='positives news rate')
    l2, = plt.plot(mps, color='green', label='stocks average')
    l3, = plt.plot(media_movel, color='midnightblue', label='positive news rate (moving average)')
    plt.legend(handles=[l1, l2, l3])
    plt.title('Positives news rate')
    plt.xlabel("Date and time")

    locs, labels = generatexTicks(interval=10 , nlocs=len(ratePos), labels=x)
    
    plt.xticks(locs, labels, rotation='90')
    plt.grid(axis='x')
    plt.ylabel("rate")
    plt.savefig("positivesNegativesNewsRates.eps", format='eps')
    plt.show()
    #plt.clf()

if __name__ == "__main__":
    names = ["negativeQtdSerie_window", "positiveQtdSerie_window"]
    windows = [1, 3, 6, 12, 24]

    for name in names:
        for window in windows:
            fname = os.path.join(source, name + "{}hours".format(window) + ".json")
            x, y = loadJson(fname)
            serie = Series(data=y, index=x)
            serie.plot(rot=45)
            # plt.set_xlabel("Quantity")
            # plt.set_ylabel("Day")
            if "negative" in name:
                title = "Negative news - "
            else:
                title = "Positive news - "
            
            title2 = "{} hour interval".format(window)
            
            plt.title(title + title2)
            plt.xlabel("Days")
            plt.ylabel("Quantity")
            # plt.show()
            plt.savefig(name + "{}hours".format(window) + ".png")
            plt.clf()

    
