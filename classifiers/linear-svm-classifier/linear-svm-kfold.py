import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Load features and labels
trainFileName = '../../features/train.txt'
with open(trainFileName, 'r') as f:
	X = f.readlines()
X = [x.strip().split(',') for x in X]
y = np.array([genres.index(x.pop()) for x in X])
X = np.array(X, dtype=float)

# Load model using pickle
modelFileName = 'linear-svm-model.pkl'
with open(modelFileName,'rb') as f:
	clf = pickle.load(f)

# KFold
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X):
	# Split data to train and test set
	# print("TRAIN: \t\t" + str(train_index))
	# print("TEST: \t\t" + str(test_index))
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	# Train
	clf.fit(X_train, y_train)

	# Print accuracy
	print("Accuracy: \t" + str(clf.score(X_test, y_test)))