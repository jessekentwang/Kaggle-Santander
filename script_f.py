import pipeline
import submitData
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

trainingFeatures, trainTarget, test=pipeline.load_data()
pipeline.digitizeMatrix(test)


print("====================Cleaning Test Data====================")
del test['index']
del test['prev_fecha_dato']
test=test.fillna(test.mean())
print("====================DONE Cleaning Test Data====================")

print("====================Filling Training Feature NA's with means...===================")
trainingFeatures.fillna(trainingFeatures.mean())
print("====================DONE Filling Training Feature NA's with means...===================")

print("===============Running PCA==================")
pca=PCA(n_components=8)
training=pca.fit_transform(trainingFeatures)
testing=pca.fit_transform(test)
print("===============DONE Runnning PCA============")

print("==============Classifiying================")
clf=GradientBoostingClassifier(n_estimators=70,verbose=True)
predictions=submitData.fitClassifier(train,lastMonthLabels,testing,clf=clf)
print("==============DONE Classifiying================")

print("PRINTING SUBMISSION")
lastMonthData=trainingFeatures[trainingFeatures.fecha_dato==15]
lastMonthLabels=trainTarget[trainingFeatures.fecha_dato==15]
submitData.printSubmission(test,trainTarget,lastMonthData,lastMonthLabels,predictions)
print("DONE!!")
