import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

total_accuracy = []
for g in genres:
	# Load data from file
	filename = 'entropy-by-genre-2/entropy-' + g + '.txt'
	with open(filename, 'r') as f:
		X = f.readlines()
	X = np.array([x.strip().split(',')[:-1] for x in X])

	# Labels
	labels = []
	for i in range(80):
		labels.append(1)
	for i in range(90):
		labels.append(0)
	y = np.array(labels)

	# Define classifier
	clf = svm.SVC(kernel='linear', C=1.0)

	# KFold
	accuracy = []
	predict = []
	kf = KFold(n_splits=10, shuffle=True)
	for train_index, test_index in kf.split(X):
		# Split data to train and test set
		# print("TRAIN: \t", train_index)
		# print("TEST: \t", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		# Train
		clf.fit(X_train, y_train)

		# Test
		# print("Actual: \t" + str(y_test))
		# print("Predicted: \t" + str(clf.predict(X_test)))

		# Print accuracy
		# print("Accuracy: \t" + str(clf.score(X_test, y_test)) + "\n")
		accuracy.append(clf.score(X_test, y_test))

		# Test
		Q = np.array([[14.240195,7.972763,53.900545,0.237402,20.198198]])
		predict.append(clf.predict(Q)[0])
	print(predict)
	print("Accuracy for " + str(g) + ": " + str(np.mean(accuracy)))
	total_accuracy.append(np.mean(accuracy))
print("Total average accuracy: " + str(np.mean(total_accuracy)))

	# Predict and test
	# print(clf.predict([[0.58, 0.76]]))
	# print(clf.predict([[10.58, 10.76]]))