from clean import *
import pandas as pd
import numpy as np
import matplotlib as plt
from pandas import DataFrame 
import pickle
import os

pdtest = pd.read_csv('../data/test_ver2.csv', delimiter = ',')

pdtest = pd.read_csv('../data/sample_submission.csv', delimiter = ',




print pdtest

i = 0
for index, row in pdtest.iterrows():
#    row['ncodpers'] = i 
 #   i = i + 1
  #  print row['ncodpers']
    
    
(rows, columns) = pdtest.shape

print rows
print columns




#[pdtain, pdtest] = cleanTrain()
