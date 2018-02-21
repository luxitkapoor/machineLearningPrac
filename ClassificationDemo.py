import sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Load data set
data = load_breast_cancer()

#Seperate dictionary keys to different sets
labelNames = data['target_names']
labels = data['target']
featureNames = data['feature_names']
features = data['data']

#Split the data set into training and testing sets
train, test, trainLabels, testLabels = train_test_split(features, labels, test_size = 0.33, random_state = 42)

#Initialize the Naive Bayes Classifier
gnb = GaussianNB()

#Train the model
model = gnb.fit(train, trainLabels)

#Use the testing data to make prediction
preds = gnb.predict(test)
print(preds)

#Calculate Accuracy
print(accuracy_score(testLabels, preds))