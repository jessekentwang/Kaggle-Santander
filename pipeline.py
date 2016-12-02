import pandas as pd
import numpy as np
from pandas import DataFrame 
import pickle
from sklearn.metrics import confusion_matrix
import os

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

def gen_classify(reg,trainFeatures,trainTarget):
	sz=int(len(trainFeatures)/2)
	cvFeatures = trainFeatures[sz:]
	trainFeatures = trainFeatures[:sz]

	conf = []
	predictions = []
	All_Targets = []

	N = len(trainTarget.columns)

	for i in range(0,N):
		target1 = trainTarget.columns[i]

		cvTarget = trainTarget[target1][sz:]
		All_Targets.append(cvTarget)

		Target = trainTarget[target1][:sz]
                print(len(Target))
                print(len(trainFeatures))
		#reg = model_method(class_weight = 'balanced')
		reg.fit(trainFeatures, Target)

		predictions.append(reg.predict(cvFeatures))

		conf.append(confusion_matrix(cvTarget, predictions))

		print (target1)
		print (conf[i])
		print (classification_report(cvTarget, predictions))
		print ('True positive rate is: ' + str((conf[i][1][1])/(conf[i][1][0] + conf[i][1][1])))
		print ('--------')

		return [predictions, cvTarget]

def load_data():
	train, test=cleanTrain()
	trainFeatures, trainTarget=split(train)
	digitizeMatrix(trainFeatures)

	return(trainFeatures,trainTarget,test)
