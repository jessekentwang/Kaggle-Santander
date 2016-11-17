import pandas as pd
import numpy as np
import matplotlib as plt
from pandas import DataFrame 
import pickle
import os

DEBUG = False
SAMPLES = 100000

def cleanTrain():
    
    print("Starting Cleaning Script!")

    if os.path.isfile('RawTrain.pickle'):
        print ("Reading Training data...")
        pdtrain = pickle.load(open(r'RawTrain.pickle', 'rb'))
        print ("Reading Test data...")
        pdtest = pickle.load(open(r'RawTest.pickle','rb'))
    else:
        print ("Reading Training data...")
        pdtest = pd.read_csv('test_ver2.csv/test_ver2.csv', delimiter = ',')
        print ("Reading Test data...")
        pdtrain = pd.read_csv('train_ver2.csv/train_ver2.csv', delimiter = ',')

    print ("done reading raw data!")
    print ("Cleaning data...")

    print (pdtrain.isnull().sum())

    if not os.path.isfile('RawTrain.pickle'):
        pdtrain = pdtrain[pdtrain['tipodom'].isnull() == False]
        pdtrain = pdtrain[pdtrain['ind_nomina_ult1'].isnull() == False]
        pdtrain = pdtrain[pdtrain['sexo'].isnull() == False]
        pdtrain = pdtrain[pdtrain['indrel_1mes'].isnull() == False]
        pdtrain = pdtrain[pdtrain['segmento'].isnull() == False]
        pdtrain = pdtrain[pdtrain['canal_entrada'].isnull() == False]
        pdtrain = pdtrain[pdtrain['cod_prov'].isnull() == False]

        stillBad = pdtrain.isnull().sum()

        for i in range(0, len(stillBad.index)):
            if stillBad[i] > 0:
                pdtrain = pdtrain.drop(stillBad.index[i], 1)
                pdtest = pdtest.drop(stillBad.index[i], 1)
    
    print ("Data Clean!")  

    print ("Writing Data...")

    if not os.path.isfile('RawTrain.pickle'):
        pickle.dump(pdtrain, open(r'RawTrain.pickle', "wb"))
        pickle.dump(pdtest, open(r'RawTest.pickle', 'wb'))

    print ("Cleaning script done!\n")

    return [pdtrain, pdtest]

if __name__ == "__main__":
    train, test = cleanTrain()
    print (train.isnull().sum())
    print (train.head())
    print (test.head())
    print (train.shape)
    print (test.shape)
