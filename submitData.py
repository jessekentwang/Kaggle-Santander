import pipeline
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import csv
import pandas as pd

def getLastMonthData():
    trainingFeatures, trainTarget, test=pipeline.load_data()
    pipeline.digitizeMatrix(test)
    print("Finding last month account statuses")
    lastMonthData=trainingFeatures[trainingFeatures.fecha_dato==16]
    lastMonthLabels=trainTarget[trainingFeatures.fecha_dato==16]
    lastMonthData= lastMonthData.groupby(lastMonthData.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
    return (lastMonthData,lastMonthLabels)




def fitClassifier(trainingFeatures,trainTarget,test,clf=RandomForestClassifier(n_estimators=60,class_weight='balanced')):
    predictions=[]
    print("Begin classification")
    N=len(trainTarget.columns)
    for i in range(0,N):
        t1=trainTarget.columns[i]
        print("Predicting "+t1)
        clf.fit(trainingFeatures,trainTarget[t1])
        predictions.append(clf.predict(test))
    print("Done with classification.")
    predictions=np.asarray(predictions).transpose()
    print(predictions)
    return predictions



def printSubmission(test,trainTarget,lastMonthData,lastMonthLabels,predictions):
    f=open('submission.csv','w')
    f.write('ncodpers,added_products\n')
    temp = lastMonthLabels.copy()
    print("Printing submission")
    i=0
    predictions_frame=pd.DataFrame(predictions,columns=temp.columns)
    predictions_frame['ncodpers'] = test['ncodpers']
    print (predictions_frame.head())

    temp['ncodpers']=lastMonthData['ncodpers']


    cols = [w+'_prev' if w!='ncodpers' else w for w in temp.columns]
    print(cols)
    temp.columns = cols
    print (temp.head())
    predictions_frame = predictions_frame.merge(temp,how='left')
    print("MERGED")
    print(predictions_frame.head())
    for w in lastMonthLabels.columns:
        if w!='ncodpers':
            predictions_frame[w]=predictions_frame[w]-predictions_frame[w+'_prev']

    for index, row in predictions_frame.iterrows():
        ret=str(int(row.ncodpers)) + ', '
        #f.write(str(row.ncodpers)+",")
        for col in lastMonthLabels.columns:
            if col!='ncodpers' and row[col]>0:
                ret+=col+' '
        #print(ret)
        f.write(ret+"\n")
    """
    lastMonthLabels=np.asarray(lastMonthLabels)
    for code in test.ncodpers:
        index_train=np.where(lastMonthData.ncodpers==code)[0]
        index_test=np.where(test.ncodpers==code)[0]
        if len(index_train)==0:
            subtracted=predictions[index_test[0]]
        else:
            subtracted=predictions[index_test[0]]-lastMonthLabels[index_train[0]]
        ret+=str(code)+', '
        for j in range(len(subtracted)):
            if subtracted[j]>0:

                ret+=trainTarget.columns[j]+' '
        ret+='\n'

    print(ret)
    f.write(ret)
    """
    f.close()
