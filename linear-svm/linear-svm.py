import numpy as np
from sklearn import svm

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
y = [0, 1, 0, 1, 0, 1]

# Define classifier
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X,y)

# Predict and test
print(clf.predict([[0.58, 0.76]]))
print(clf.predict([[10.58, 10.76]]))