#Feature engineering for Titanic ML contest
#Implementation of Scikit learn SVM, Logistic regression and Random forest models
#Author : Arjun Chakraborty


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import re


def fix_data(df):

	#  Fixing the training data
	df["Age"] = df["Age"].fillna(df["Age"].median())
	df.loc[df["Sex"] == "male", "Sex"] = 0
	df.loc[df["Sex"] == "female", "Sex"] = 1

	#  Filling up embarked with S
	df["Embarked"].fillna("S")
	df.loc[df["Embarked"] == "S", "Embarked"] = 0
	df.loc[df["Embarked"] == "C", "Embarked"] = 1
	df.loc[df["Embarked"] == "Q", "Embarked"] = 2

	#  Filling up fare with median
	df["Fare"] = df["Fare"].fillna(df["Fare"].median())


	#  Creating a new cabinletter feature

	# Replace missing values with "U0"
	df['Cabin'][df.Cabin.isnull()] = 'U0'

	# create feature for the alphabetical part of the cabin number
	df['CabinLetter'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())

	# convert the distinct cabin letters with incremental integer values
	df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]

	# Create a feature for the deck
	df['Deck'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
	df['Deck'] = pd.factorize(df['Deck'])[0]
	 
	# Create binary features for each deck
	decks = pd.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
	df = pd.concat([df, decks], axis=1)
	 
	# Create feature for the room number
	# df['Room'] = df['Cabin'].map( lambda x : re.compile("([0-9]+)").search(x).group()).astype(int) + 1




	return df


def predictTitanic(train,test,predictors):

	#  Predictions using logistic regression

	logistic = linear_model.LogisticRegression()
	logistic.fit(train[predictors],train["Survived"])
	print " THe score for logistic regression is"
	print logistic.score(train[predictors],train["Survived"])


	# Predictions using scikit learn svm

	clf = svm.SVC()
	clf.fit(train[predictors],train["Survived"])
	print "THe score for SVM is"
	print clf.score(train[predictors],train["Survived"])
	predictions_svm = clf.predict(test[predictors])

	#Predictions using random forest models
	numEstimators = 100
 	model = RandomForestClassifier(numEstimators)
  	model.fit(train[predictors],train["Survived"])
	print "THe score for RF is"
	print model.score(train[predictors],train["Survived"])
	predictions_RFM = model.predict(test[predictors])


	submission = pd.DataFrame({
       "PassengerId": test["PassengerId"],
       "Survived": predictions_RFM
      })

	submission.to_csv('submission_RFM.csv', index=False)






train = pd.read_csv('/Users/arjun/Documents/Titanic/train.csv', dtype={"Age": np.float64})
test = pd.read_csv('/Users/arjun/Documents/Titanic/test.csv', dtype={"Age": np.float64})

train = fix_data(train)
test = fix_data(test)
predictors = ["Pclass","Sex", "Age","SibSp","Fare","CabinLetter","Deck"]

predictTitanic(train,test,predictors)







