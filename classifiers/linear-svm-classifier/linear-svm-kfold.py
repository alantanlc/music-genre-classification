import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Load features and labels
filename = '..\\..\\features\\combined.txt'
with open(filename, 'r') as f:
	X = f.readlines()
X = [x.strip().split(',') for x in X]
y = np.array([genres.index(x.pop()) for x in X])
X = np.array(X, dtype=float)

# Define classifier
clf = svm.SVC(kernel='linear', C=1.0)

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

	# Test
	# print("Actual: \t" + str(y_test))
	# print("Predicted: \t" + str(clf.predict(X_test)))

	# Print accuracy
	print("Accuracy: \t" + str(clf.score(X_test, y_test)))

	# Example usage to test an instance
	# print(clf.predict([[0, 10]]))