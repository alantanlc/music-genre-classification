import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

# Genres
genres_1 = ['blues','classical','country','hiphop']
genres_2 = ['jazz','metal','pop','disco','reggae']
genres_3 = ['rock','reggae']

# Genres 1
X_1 = []
for g in genres_1:
	# Load features and labels
	trainFileName = './features/features-' + g + '.arff'
	with open(trainFileName, 'r') as f:
		X = f.readlines()
		X_1 = X_1 + X
X_1 = [x.strip().split(',') for x in X_1]
y_1 = np.array([genres_1.index(x.pop()) for x in X_1])
X_1= np.array(X_1, dtype=float)

# Genres 2
X_2 = []
for g in genres_2:
	# Load features and labels
	trainFileName = './features/features-' + g + '.arff'
	with open(trainFileName, 'r') as f:
		X = f.readlines()
		X_2 = X_2 + X
X_2 = [x.strip().split(',') for x in X_2]
y_2 = np.array([genres_2.index(x.pop()) for x in X_2])
X_2= np.array(X_2, dtype=float)

# Genres 3
X_3 = []
for g in genres_3:
	# Load features and labels
	trainFileName = './features/features-' + g + '.arff'
	with open(trainFileName, 'r') as f:
		X = f.readlines()
		X_3 = X_3 + X
X_3 = [x.strip().split(',') for x in X_3]
y_3 = np.array([genres_3.index(x.pop()) for x in X_3])
X_3= np.array(X_3, dtype=float)

# Define classifier
clf = svm.SVC(kernel='linear', C=1.0)


# KFold 1
accuracy = []
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X_1):
	# Split data to train and test set
	X_train, X_test = X_1[train_index], X_1[test_index]
	y_train, y_test = y_1[train_index], y_1[test_index]

	# Train
	clf.fit(X_train, y_train)

	# Print accuracy
	acc = clf.score(X_test, y_test)
	accuracy.append(acc)
	print("Accuracy: \t" + str(acc))

# Print average accuracy
print("Total Average Accuracy for Genres Group 1: \t" + str(np.mean(accuracy)) + '\n')


# KFold 2
accuracy = []
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X_2):
	# Split data to train and test set
	X_train, X_test = X_2[train_index], X_2[test_index]
	y_train, y_test = y_2[train_index], y_2[test_index]

	# Train
	clf.fit(X_train, y_train)

	# Print accuracy
	acc = clf.score(X_test, y_test)
	accuracy.append(acc)
	print("Accuracy: \t" + str(acc))

# Print average accuracy
print("Total Average Accuracy for Genres Group 2: \t" + str(np.mean(accuracy)) + '\n')


# KFold 3
accuracy = []
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X_3):
	# Split data to train and test set
	X_train, X_test = X_3[train_index], X_3[test_index]
	y_train, y_test = y_3[train_index], y_3[test_index]

	# Train
	clf.fit(X_train, y_train)

	# Print accuracy
	acc = clf.score(X_test, y_test)
	accuracy.append(acc)
	print("Accuracy: \t" + str(acc))

# Print average accuracy
print("Total Average Accuracy for Genres Group 3: \t" + str(np.mean(accuracy)) + '\n')