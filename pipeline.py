import pandas as pd
import numpy as np
from pandas import DataFrame
import pickle
from sklearn.metrics import confusion_matrix
import average_precision
import os
import matplotlib.pyplot as plt
from numpy.linalg import svd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import featuretransform
from featuretransform import timetransform

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
		for x in allData2.columns:
			if x == 'fecha_dato' or x == 'ncodpers' or fits(x):
				continue
			del allData2[x]

		print ('Dataframes made!')
		print (allData.head())

		allData2.rename(columns = lambda x: prevName(x), inplace = True)

		print ('DF2 renamed!')
		print (allData2.head())

		a = set(allData['fecha_dato'])
		allData['fecha_dato_prev'] = allData['fecha_dato'].apply(lambda x: prevDate(x, a))

		print ('DF1 dates readjusted')
		print ('Now merging...')

		retval = pd.merge(allData, allData2, how = 'left', on = ['fecha_dato_prev', 'ncodpers'])
		retval['total_accounts_open'] = 0
		for x in retval.columns:
			if len(x) > 9 and x[:4] == 'ind_' and x[-5:] == '_prev':
				retval['total_accounts_open'] = retval.total_accounts_open + retval[x]

		return retval

def cleanTrain(n = None):

	print("Starting Cleaning Script!\n")


	if os.path.isfile('RawTrain.pickle'):
		print ("Reading Training data...")
		pdtrain = pickle.load(open(r'RawTrain.pickle', 'rb'))
		print ("Reading Test data...")
		pdtest = pickle.load(open(r'RawTest.pickle','rb'))
	else:
		print ("Reading Training data...")
		pdtest = pd.read_csv('./test_ver2.csv', delimiter = ',')
		print ("Reading Test data...")
		pdtrain = pd.read_csv('./test_ver2.csv', delimiter = ',')

		#print('old shape: ' + str(pdtrain.shape))

		#print(str(pdtrain.shape))

		alldata = addFeatures([pdtrain, pdtest])
		alldata['age'] = pd.to_numeric(alldata.age, errors = 'coerce')
		alldata['antiguedad'] = pd.to_numeric(alldata.antiguedad, errors = 'coerce')
		alldata['indrel_1mes'] = pd.to_numeric(alldata.indrel_1mes, errors = 'coerce')
		alldata['conyuemp'].fillna(0, inplace = True)
		alldata['ult_fec_cli_1t'].fillna(0, inplace = True)
		alldata['tipodom'].fillna(0, inplace = True)
		alldata = alldata[alldata.total_accounts_open.isnull() == False]



		"""for x in alldata.columns:
			print (x)
			if not fits(x):
				mean_x = alldata[x].mean()
				alldata[x].fillna(mean_x, inplace = True)"""

		pdtrain = alldata[alldata.fecha_dato != '2016-06-28']
		pdtest = alldata[alldata.fecha_dato == '2016-06-28']
		pdtrain = pdtrain[pdtrain['ind_viv_fin_ult1_prev'].isnull() == False]
		#TODO: Why does this line drop everything?
		pdtest = pdtest.dropna(axis = 1, how = 'all')

		print ("done reading raw data!")
		print ("Cleaning data...")

		print (pdtrain.isnull().sum())

		if not os.path.isfile('RawTrain.pickle'):
			#pdtrain = pdtrain[pdtrain['tipodom'].isnull() == False]
			pdtrain = pdtrain[pdtrain['ind_nomina_ult1'].isnull() == False]
			"""pdtrain = pdtrain[pdtrain['sexo'].isnull() == False]
			pdtrain = pdtrain[pdtrain['indrel_1mes'].isnull() == False]
			pdtrain = pdtrain[pdtrain['segmento'].isnull() == False]
			pdtrain = pdtrain[pdtrain['canal_entrada'].isnull() == False]
			pdtrain = pdtrain[pdtrain['cod_prov'].isnull() == False]"""

			"""stillBad = pdtrain.isnull().sum()

			for i in range(0, len(stillBad.index)):
				if stillBad[i] > 0:
					pdtrain = pdtrain.drop(stillBad.index[i], 1)
					pdtest = pdtest.drop(stillBad.index[i], 1)"""

		print ("Data Clean!")

		print ("Writing Data...")

		pdtrain = pdtrain.reset_index()
		pdtest = pdtest.reset_index()

		"""if not os.path.isfile('RawTrain.pickle'):
			pickle.dump(pdtrain, open(r'RawTrain.pickle', "wb"))
			pickle.dump(pdtest, open(r'RawTest.pickle', 'wb'))"""

		#print ("pdtrain shape: ", pdtrain.shape)
		#print ("pdtest shape: ", pdtest.shape)

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

def gen_classify_cv(reg,trainFeatures,trainTarget):
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
		reg.fit(trainFeatures, Target)

		predictions.append(reg.predict(cvFeatures))

		conf.append(confusion_matrix(cvTarget, predictions[i]))

		print (target1)
		print (conf[i])
		#print (classification_report(cvTarget, predictions[i]))
		print ('True positive rate is: ' + str(float(conf[i][1][1])/float(conf[i][1][0] + conf[i][1][1])))
		print ('Accuracy is: ' + str(float(conf[i][0][0] + conf[i][1][1])/float(conf[i][0][0]+ conf[i][0][1] + conf[i][1][0] + conf[i][1][1])))
		print ('--------')

	return [predictions, All_Targets]

def replaceNanWithMean(matrix):
	averages=matrix.mean()
	for col in matrix.columns:
		if col in averages.keys():
			matrix[col].fillna(averages[col])

def getPCAVariances(matrix):
	pca=PCA()
	pca.fit(matrix)
	return pca.explained_variance_

def gen_classify_test(reg,trainFeatures,trainTarget,month,runPCA=False):
	trainingData=trainFeatures[trainFeatures.fecha_dato==month]
	trainingLabels=trainTarget[trainFeatures.fecha_dato==(month)]
	testingData=trainFeatures[trainFeatures.fecha_dato==(month+1)]
	testingLabels=trainTarget[trainFeatures.fecha_dato==(month+1)]
	predictions=[]
	conf=[]
	if(runPCA!=False):
		pca=PCA(n_components=runPCA)
		trainingData=pca.fit_transform(trainingData)
		testingData=pca.fit_transform(testingData)
	N=len(testingLabels.columns)
	for i in range(0,N):
		t1=(trainingLabels.columns[i])
		reg.fit(trainingData,trainingLabels[t1])
		predictions.append(reg.predict(testingData))
		conf.append(confusion_matrix(testingLabels[t1],predictions[i]))

		print(t1)
		print(conf[i])
		print ('True positive rate is: ' + str(float(conf[i][1][1])/float(conf[i][1][0] + conf[i][1][1])))
		print("Accuracy is: "+str(float(conf[i][0][0]+conf[i][1][1])/float(conf[i][0][0]+conf[i][0][1]+conf[i][1][0]+conf[i][1][1])))
		print ('--------')
	print("AVERAGE PRECISION: ")
	print(average_precision.mapk(np.asarray(testingLabels),np.asarray(predictions)))
	return [predictions,conf]

def load_data():
	train, test=cleanTrain()
	train = timetransform(train)
	trainFeatures, trainTarget=split(train)
	digitizeMatrix(trainFeatures)
	del trainFeatures['fecha_dato_prev']
	trainFeatures = trainFeatures[trainFeatures['fecha_dato'] != 0]

	return(trainFeatures,trainTarget,test)

if __name__ == "__main__":
	data = load_data()
	model = RandomForestClassifier(n_estimators = 50, class_weight = 'balanced', verbose = 1)
	result = gen_classify_test(model, data[0], data[1], 5)
