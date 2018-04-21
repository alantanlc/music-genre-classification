import pickle
import numpy as np
from sklearn import svm

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Load training features and labels
trainFileName = '../../features/train-features.txt'
with open(trainFileName, 'r') as f:
	trainX = f.readlines()
trainX = [x.strip().split(',') for x in trainX]
trainY = np.array([genres.index(x.pop()) for x in trainX])
trainX = np.array(trainX, dtype=float)

# Load datasetTest
testFileName = '../../features/predict-features.txt'
with open(testFileName, 'r') as f:
	predictX = f.readlines()
predictX = [x.strip().split(',') for x in predictX]
predictX = np.array(predictX, dtype=float)

# Define classifier
modelFileName = 'linear-svm-model.pkl'
with open(modelFileName,'rb') as f:
	clf = pickle.load(f)

# Predict datasetTest
predictY = clf.predict(predictX)
predictY = [genres[y]+'\n' for y in predictY]

# Write to file
predictFileName = 'predictions.txt'
with open(predictFileName, 'w') as f:
	f.writelines(predictY)

print('Program completed!')