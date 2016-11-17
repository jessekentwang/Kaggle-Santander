print "DO RANDOM FOREST"

Jupyter = False

# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame


pd.set_option("display.max_columns", 100)

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

if Jupyter:
    print "boo"#matplotlib inline

# machine learning
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB

# get titanic & test csv files as a DataFrame
train_df = pd.read_csv("../../data/TitanicTutorial/train.csv")
test_df    = pd.read_csv("../../data/TitanicTutorial/test.csv")


print("----------------------------")
train_df.info()
test_df.info()
print("----------------------------")


# define training and testing sets

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()




# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)



submission = pd.DataFrame({
       # "PassengerId": test_df["PassengerId"],
       # "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

