import pickle
import numpy as np
from sklearn import svm

# Genres
genres_1 = ['blues','classical','country','hiphop']
genres_2 = ['jazz','metal','pop','disco','reggae']
genres_3 = ['rock','reggae']
genres_for_rock = ['blues','classical','country','hiphop','jazz','metal','pop','disco','reggae']

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

# Train and save model 1
print('Train and save model 1...')
print(genres_1)
clf.fit(X_1, y_1)
modelFileName = 'models/linear-svm-model-blues-classical-country-hiphop.pkl'
with open(modelFileName, 'wb') as f:
	pickle.dump(clf, f)

# Train and save model 2
clf.fit(X_2, y_2)
print('Train and save model 2...')
print(genres_2)
modelFileName = 'models/linear-svm-model-jazz-metal-pop-disco-reggae.pkl'
with open(modelFileName, 'wb') as f:
	pickle.dump(clf, f)

# Train and save rock models
for g_R in genres_for_rock:
	# Genres
	X_R = []
	genres = ['rock', g_R]
	for g in genres:
		# Load features and labels
		trainFileName = './features/features-' + g + '.arff'
		with open(trainFileName, 'r') as f:
			X = f.readlines()
			X_R = X_R + X
	X_R = [x.strip().split(',') for x in X_R]
	y_R = np.array([genres.index(x.pop()) for x in X_R])
	X_R = np.array(X_R, dtype=float)
	# Train and save model
	print('Train and save rock-' + g_R + ' model...')
	clf.fit(X_R, y_R)
	modelFileName = 'models/linear-svm-model-rock-' + g_R + '.pkl'
	with open(modelFileName, 'wb') as f:
		pickle.dump(clf, f)

# Train and save genres_1-genres_2 combinations
for g1 in genres_1:
	for g2 in genres_2:
		# Genres
		X_R = []
		genres = [g1, g2]
		for g in genres:
			# Load features and labels
			trainFileName = './features/features-' + g + '.arff'
			with open(trainFileName, 'r') as f:
				X = f.readlines()
				X_R = X_R + X
		X_R = [x.strip().split(',') for x in X_R]
		y_R = np.array([genres.index(x.pop()) for x in X_R])
		X_R = np.array(X_R, dtype=float)
		# Train and save model
		print('Train and save ' + g1 + '-' + g2 + ' model...')
		clf.fit(X_R, y_R)
		modelFileName = 'models/linear-svm-model-' + g1 + '-' + g2 + '.pkl'
		with open(modelFileName, 'wb') as f:
			pickle.dump(clf, f)

print('Program completed!')