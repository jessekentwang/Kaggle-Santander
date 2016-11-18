import pandas as pd
import numpy as np
from pandas import DataFrame 


def digitizeMatrix(raw_dataframe):
    col_names=raw_dataframe.columns.values
    for col_name in col_names:
        if isinstance(raw_dataframe[col_name][0],str):
            raw_dataframe[col_name]=pd.Categorical(raw_dataframe[col_name]).codes
    

if __name__ == '__main__':
    print("Reading data...")
    pdtest = pd.read_csv('test_ver2.csv', delimiter = ',')
    #rawtraining=DataFrame.as_matrix(pdtest)
    print("Done!")
    digitizeMatrix(pdtest)
    print(pdtest)
                    
                    
