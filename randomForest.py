print "DO RANDOM FOREST"

# Imports

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clean.py import *
from sklearn.ensemble import RandomForestClassifier
sns.set_style('whitegrid')



# get titanic & test csv files as a DataFrame
[pdtrain, pdtest] = cleanTrain()
print("----------------------------")
pdtrain.info()
pdtest.info()
print("----------------------------")


# define training and testing sets

X_train =  #titanic_df.drop("Survived",axis=1)
Y_train =  # titanic_df["Survived"]
X_test  =  #test_df.drop("PassengerId",axis=1).copy()

# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)


added_products = pd.zeros(len(pdtest[0]))

submission = pd.DataFrame({
       "ncodpers": pdtest["ncodpers"],
       "added_products": Y_pred
    })
submission.to_csv('submissionRF.csv', index=False)

