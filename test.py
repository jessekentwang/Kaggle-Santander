import pandas as pd
import numpy as np
import matplotlib as plt
from pandas import DataFrame 
import pickle

DEBUG = False
SAMPLES = 100000

def cleanTrain():
    if not DEBUG:
        pdtest = pd.read_csv('test_ver2.csv/test_ver2.csv', delimiter = ',')
        pdtrain = pd.read_csv('train_ver2.csv/train_ver2.csv', delimiter = ',')


        pickle.dump(pdtrain, open(r'RawTrain.pickle', "wb"))
        pickle.dump(pdtest, open(r'RawTest.pickle', 'wb'))

    if DEBUG:
        train = pd.read_csv('train_ver2.csv/train_ver2.csv', delimiter = ',')
        train2 = DataFrame.as_matrix(train)

    return pdtrain
    
train = cleanTrain()

