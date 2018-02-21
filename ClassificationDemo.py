import sklearn

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

labelNames = data['target_names']
labels = data['target']
featureNames = data['feature_names']
features = data['data']

from sklearn.model_selection import train_test_split

train, test, trainLabels, testLabels = train_test_split(features, labels, test_size = 0.33, random_state = 42)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit(train, trainLabels)

preds = gnb.predict(test)
print(preds)

from sklearn.metrics import accuracy_score

print(accuracy_score(testLabels, preds))