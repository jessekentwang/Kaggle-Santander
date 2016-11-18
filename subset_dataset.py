# Take a percentage of the original data set for faster algorithmic testing


import pandas as pd
import numpy as np
import matplotlib as plt
from pandas import DataFrame 
import pickle
import os

DEBUG = False
SAMPLES = 100000

def subsetTrain():
    
    print("Starting subsetting ....")

    if not os.path.isfile('SubsetTrain.pickle') or not os.path.isfile('SubsetTest.pickle'):
        
        if os.path.isfile('RawTrain.pickle'):
            print ("Reading Training data from pickle file...")
            pdtrain = pickle.load(open(r'RawTrain.pickle', 'rb'))
            print ("Reading Test data...")
            pdtest = pickle.load(open(r'RawTest.pickle','rb'))
        else:
            print ("Reading Training data from csv...")
            pdtest = pd.read_csv('../data/test_ver2.csv', delimiter = ',')
            print ("Reading Test data...")
            pdtrain = pd.read_csv('../data/train_ver2.csv', delimiter = ',')
            
            print ("done reading raw data!")
            print ("Subseting data...")
            
            print (pdtrain.isnull().sum())
            
            if not os.path.isfile('SubsetTrain.pickle'):
                random.seed(1337)
                trainvec = range(len(pdtrain))
                testvec = range(len(pdtest))
                train_filter = random.sample(trainvec, len(pdtrain)/100)
                test_filter = random.sample(testvec, len(pdtest)/100)
                pdtrain = pdtrain[train_filter]
                pdtest = pdtest[test_filter]
            
            print ("Raw Data Subset!")  
            
            print ("Writing Data...")
            
            if not os.path.isfile('SubsetTrain.pickle'):
                pickle.dump(pdtrain, open(r'SubsetTrain.pickle', "wb"))
                pickle.dump(pdtest, open(r'SubsetTest.pickle', 'wb'))
                
                print ("Subsetting script done!\n")


    print ("Reading Subset data from pickle file...")
    pdtrain = pickle.load(open(r'SubsetTrain.pickle', 'rb'))
    print ("Reading Subset Test data from pickle file ...")
    pdtest = pickle.load(open(r'SubsetTest.pickle','rb'))

    print "subset shape: ", pdtrain.shape
    print "test subset shape: ", pdtest.shape
    
    return [pdtrain, pdtest]

if __name__ == "__main__":
    train, test = subsetTrain()
    print (train.isnull().sum())
    print (train.head())
    print (test.head())
    print (train.shape)
    print (test.shape)
