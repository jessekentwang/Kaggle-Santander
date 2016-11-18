import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle

def fits(c):
    if len(c) > 8 and c[:4] == 'ind_' and c[-5:] == '_ult1':
        return True
    return False

def split(data):
    target_cols = [c for c in data.columns if fits(c)]
    feature_cols = [c for c in data.columns if c not in target_cols]

    print (target_cols)
    print (feature_cols)

    return [data[feature_cols], data[target_cols]]

def upsample(data, col_name):
    pass
