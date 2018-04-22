import pickle
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Load datasetTest
testFileName = '../../features/predict-features.txt'
with open(testFileName, 'r') as f:
	predictX = f.readlines()
predictX = [x.strip().split(',') for x in predictX]
predictX = np.array(predictX, dtype=float)

# Load scaler
scalerFileName = 'scaler-preprocessed.pkl'
with open(scalerFileName,'rb') as f:
	scaler = pickle.load(f)
predictX = scaler.transform(predictX)

# Load classifier
modelFileName = 'linear-svm-model-preprocessed.pkl'
with open(modelFileName,'rb') as f:
	clf = pickle.load(f)

# Predict datasetTest
predictY = clf.predict(predictX)
predictY = [genres[y]+'\n' for y in predictY]

# Write to file
predictFileName = 'predictions-preprocessed.txt'
with open(predictFileName, 'w') as f:
	f.writelines(predictY)

print('Program completed!')