import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

# Feature list
X = np.array([
	[1, 1],
	[1, 2],
	[2, 1],
	[2, 2],
	[8, 8],
	[8, 9],
	[9, 8],
	[9, 9],
	[1, 8],
	[1, 9],
	[2, 8],
	[2, 9]
])

# Labels
y = np.array([1,1,1,1,2,2,2,2,3,3,3,3])

# Define classifier
clf = svm.SVC(kernel='linear', C=1.0)

# KFold
kf = KFold(n_splits=2, shuffle=True)
for train_index, test_index in kf.split(X):
	# Split data to train and test set
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	# Train
	clf.fit(X_train, y_train)

	# Test
	print("Actual Labels: " + str(y_test))
	print("Predicted Labels: " + str(clf.predict(X_test)))

	# Print accuracy
	print("Accuracy: " + str(clf.score(X_test, y_test)))
	print(clf.predict([[0, 10]]))
	print("")

# Predict and test
# print(clf.predict([[0.58, 0.76]]))
# print(clf.predict([[10.58, 10.76]]))