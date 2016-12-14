import submitData
import pipeline
from sklearn.decomposition import PCA

lastMonthData=trainingFeatures[trainingFeatures.fecha_dato>=10]
lastMonthLabels=trainTarget[trainingFeatures.fecha_dato>=10]
lastMonthData=lastMonthData.fillna(lastMonthData.mean())



test=test.fillna(test.mean())
pca=PCA(n_components=8)

training=pca.fit_transform(lastMonthData)
testing=pca.fit_transform(test)

predictionss=submitData.fitClassifier(train,lastMonthLabels,testing,clf=clf)
submitData.printSubmission(test,trainTarget,lastMonthData,lastMonthLabels,predictionss)
