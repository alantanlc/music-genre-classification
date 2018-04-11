import sys
import unidecode
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

# Get filename from command line argument
filename = str(sys.argv[1])

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
y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

# Define classifier
clf = svm.SVC(kernel='linear', C=1.0)

# KFold
kf = KFold(n_splits=2, shuffle=True)
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
	# print("Accuracy: \t" + str(clf.score(X_test, y_test)) + "\n")

	# Example usage to test an instance
	# print(clf.predict([[0, 10]]))

# Return results to Node server
result = str(clf.predict([[0, 10]])) + ", " + filename
print(unidecode.unidecode_expect_nonascii(result))
sys.stdout.flush()