import pandas as pd
import numpy as np
from pandas import DataFrame 
import pickle
from sklearn.metrics import confusion_matrix
import os

def prevDate(date, a):
        if date.startswith('2015-01'):
            return None
        elif date.startswith('2016-01'):
            return next((x for x in a if x.startswith('2015-12')), None)
        else:
            ltmp = date[:5]
            tmp = (str(int(date[5:7]) - 1)).rjust(2, '0')
            return next((x for x in a if x.startswith(ltmp+tmp)), None) 

def prevName(name):
        if fits(name) or name == 'fecha_dato':
            return name + '_prev'
        else:
            return name

def addFeatures(data):
        train = data[0]
        test = data[1]

        allData = pd.concat([train,test])
        allData2 = allData.copy()

        print ('Dataframes made!')
        print (allData.head())

        allData2 = allData2.rename(columns = lambda x: prevName(x), inplace = True)

        print ('DF2 renamed!')
        print (allData.head())

        a = set(allData['fecha_dato'])
        allData['fecha_dato_prev'] = allData['fecha_dato'].apply(lambda x: prevDate(x, a))

        print ('DF1 dates readjusted')
        print ('Now merging...')

        allData = pd.merge(allData, allData2, how = 'left', on = ['fecha_dato_prev', 'ncodpers'])

        return allData

def cleanTrain(n = None):
	
	print("Starting Cleaning Script!\n")

	if os.path.isfile('RawTrain.pickle'):
		print ("Reading Training data...")
		pdtrain = pickle.load(open(r'RawTrain.pickle', 'rb'))
		print ("Reading Test data...")
		pdtest = pickle.load(open(r'RawTest.pickle','rb'))
	else:
		print ("Reading Training data...")
		pdtest = pd.read_csv('../data/test_ver2.csv', delimiter = ',')
		print ("Reading Test data...")
		pdtrain = pd.read_csv('../data/train_ver2.csv', delimiter = ',')

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

	print ("pdtrain shape: ", pdtrain.shape)  
	print ("pdtest shape: ", pdtest.shape)
		
	print ("Cleaning script done!\n")

	
	
	return [pdtrain, pdtest]

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

def digitizeMatrix(raw_dataframe):
	col_names=raw_dataframe.columns.values
	for col_name in col_names:
		if isinstance(raw_dataframe[col_name][0],str):
			raw_dataframe[col_name]=pd.Categorical(raw_dataframe[col_name]).codes

def gen_classify(reg):
	train, test = cleanTrain()

	trainFeatures, trainTarget = split(train)
	digitizeMatrix(trainFeatures)

	cvFeatures = trainFeatures[-6500000:]
	trainFeatures = trainFeatures[:-6500000]

	conf = []
	predictions = []
	All_Targets = []

	N = len(trainTarget.columns)

	for i in range(0,N):
		target1 = trainTarget.columns[i]

		cvTarget = trainTarget[target1][-6500000:]
		All_Targets.append(cvTarget)

		Target = trainTarget[target1][:-6500000]

		#reg = model_method(class_weight = 'balanced')
		reg.fit(trainFeatures, Target)

		predictions.append(reg.predict(cvFeatures))

		conf.append(confusion_matrix(cvTarget, predictions[i]))

		print (target1)
		print (conf[i])
		#print (classification_report(cvTarget, predictions[i]))
		print ('True positive rate is: ' + str((conf[i][1][1])/(conf[i][1][0] + conf[i][1][1])))
		print ('--------')

	return [predictions, cvTarget]

def load_data():
	train, test=cleanTrain()
	trainFeatures, trainTarget=split(train)
	digitizeMatrix(trainFeatures)

	return(trainFeatures,trainTarget,test)

if __name__ == "__main__":
        a = cleanTrain()
        print (a[0].head())

