import numpy as np
from sklearn import svm

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Load training features and labels
trainFileName = '..\\..\\datasetTest\\combined.txt'
with open(trainFileName, 'r') as f:
	trainX = f.readlines()
trainX = [x.strip().split(',') for x in trainX]
trainY = np.array([genres.index(x.pop()) for x in trainX])
trainX = np.array(trainX, dtype=float)

# Load datasetTest
testFileName = '..\\..\\features\\datasetTest.txt'
with open(testFileName, 'r') as f:
	predictX = f.readlines()
predictX = [x.strip().split(',') for x in predictX]
predictX = np.array(predictX, dtype=float)

# Define classifier
clf = svm.SVC(kernel='linear', C=1.0)

# Train classifier
clf.fit(trainX, trainY)

# Predict datasetTest
predictY = clf.predict([predictX])

# Write to file
predictFileName = 'predict.txt'
with open(predictFileName, 'w') as f:
	f.writelines(predictY)
