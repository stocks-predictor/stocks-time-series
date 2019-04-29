import json
import os
import datetime
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

    plt.figure(figsize=(20, 15))
    plt.plot(x, ratePos, color='blue')
    plt.plot(mps, color='green')
    plt.title('Positives news rate')
    plt.xlabel("Date and time")

    locs, labels = generatexTicks(interval=2 , nlocs=len(ratePos), labels=x)
    
    plt.xticks(locs, labels, rotation='90')
    plt.grid(axis='x')
    plt.ylabel("rate")
    #plt.show()
    plt.savefig("positivesNegativesNewsRates.eps", format='eps')
    plt.clf()

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

    