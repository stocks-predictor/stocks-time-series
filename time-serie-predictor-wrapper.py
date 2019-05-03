# -*- encoding: utf-8 -*-

from pandas_datareader import data
import pandas as pd
import datetime as dt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='time series stocks predictor')
parser.add_argument('-i', '--input', type=str, default= 'utils/dataSeriePosNeg.json', help='dataset path')
parser.add_argument('-e', '--epochs', type=int, default= 5, help='number of epochs to fit the model')
parser.add_argument('-lb', '--look_back', type=int, default= 50, help='how many previous time the lstm see')

####################################GLOBAL VARIABLES#############################################################
#test_set_size= 50
PATH_IN = parser.parse_args().input
epochs = parser.parse_args().epochs
look_back = parser.parse_args().look_back
#################################################################################################################
