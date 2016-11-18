#import numpy as np
#import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import svm
#import os
from split import *
from clean import *
from cleaning import *
from sklearn.metrics import classification_report

train, test = cleanTrain()

trainFeatures, trainTarget = split(train)
digitizeMatrix(trainFeatures)

cvFeatures = trainFeatures[-6500000:]
trainFeatures = trainFeatures[:-6500000]

conf = []

N = len(trainTarget.columns)

for i in range(0,N):
    target1 = trainTarget.columns[i]

    cvTarget = trainTarget[target1][-6500000:]

    Target = trainTarget[target1][:-6500000]

    reg = linear_model.LogisticRegression(class_weight = 'balanced')
    reg.fit(trainFeatures, Target)

    predictions = reg.predict(cvFeatures)

    conf[i] = confusion_matrix(cvTarget, predictions)

    print (target1)
    print (conf[i])
    print (classification_report(cvTarget, predictions))
    print ('True positive rate is: ' + str((conf[i][1][1])/(conf[i][1][0] + conf[i][1][1])))
    print ('--------')
