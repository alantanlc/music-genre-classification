import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

# Feature list
X = np.array([
	[1, 2],
	[5, 8],
	[1.5, 1.8],
	[8, 8],
	[1, 0.6],
	[9, 11]
])

# Labels
y = np.array([0, 1, 0, 1, 0, 1])

# Define classifier
clf = svm.SVC(kernel='linear', C=1.0)

# KFold
kf = KFold(n_splits=2, shuffle=True)
for train_index, test_index in kf.split(X):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	# Train
	clf.fit(X_train, y_train)

	# Test
	print(clf.predict(X_test))

# Predict and test
# print(clf.predict([[0.58, 0.76]]))
# print(clf.predict([[10.58, 10.76]]))