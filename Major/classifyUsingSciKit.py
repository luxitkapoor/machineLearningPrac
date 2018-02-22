import pandas 
import numpy
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm

filename = "Country_A_Household_Train.csv"
data = pandas.read_csv(filename, index_col='id')

#data.poor.value_counts().plot.bar(title = 'number of poor')




def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    

    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    
    return df
    

def pre_process_data(df, enforce_cols=None):

        

    df = standardize(df)

        

    df = pandas.get_dummies(df)

   
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(0, inplace=True)
    
    return df

dataX = pre_process_data(data.drop('poor', axis=1))
dataY = numpy.ravel(data.poor)
# dataX.to_csv('normalData.csv')

def naiveBayesClassifier():
	train, test, trainLabels, testLabels = train_test_split(dataX, dataY, test_size = 0.33, random_state = 42)

	gnb = GaussianNB()
	model = gnb.fit(train, trainLabels)
	preds = gnb.predict(test)
	return accuracy_score(testLabels, preds)


def supportVector():
	train, test, trainLabels, testLabels = train_test_split(dataX, dataY, test_size = 0.33, random_state = 42)
	classifier = svm.SVC()
	model = classifier.fit(train, trainLabels)
	preds = classifier.predict(test)

	return accuracy_score(testLabels, preds)

accuracies = []
accuracies.append(naiveBayesClassifier())
accuracies.append(supportVector())
labels = ["Naive Bayes Accuracy", "Support Vector Accuracy"]
index = numpy.arange(len(labels))
plt.bar(index, accuracies)
plt.xticks(index, labels)
plt.title("Accuracy")
plt.show()