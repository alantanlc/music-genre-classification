import pickle
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Load features and labels
fileName = '../../features/train-features.txt'
with open(fileName, 'r') as f:
	X = f.readlines()
X = [x.strip().split(',') for x in X]
y = np.array([genres.index(x.pop()) for x in X])
X = np.array(X, dtype=float)


# train test split
#X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train=X
#y_train=y

# preprocessing phase
scaler = StandardScaler()
scaler.fit(X_train)
# StandardScaler(copy=True, with_mean=True, with_std=True)

# apply the transformations to the data:
X_train = scaler.transform(X_train)



# Define classifier
clf = svm.SVC(kernel='linear', C=1.0)

# Train
clf.fit(X, y)

# Save model using pickle
modelFileName = 'linear-svm-model.pkl'
with open(modelFileName, 'wb') as f:
	pickle.dump(clf, f)

# Load model using pickle
# modelFileName = 'linear-svm-model.pkl'
# with open(modelFileName,'rb') as f:
# 	clf = pickle.load(f)

print('Program completed!')