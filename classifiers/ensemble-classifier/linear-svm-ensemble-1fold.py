import pickle
import numpy as np

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
genres_1 = ['blues','classical','country','hiphop']
genres_2 = ['jazz','metal','pop','disco','reggae']

# K Fold Directory Name
kFoldDirName = 'kfold-9'

# Load model 1 using pickle
modelFileName = './kfold-models/' + kFoldDirName + '/linear-svm-model-blues-classical-country-hiphop.pkl'
with open(modelFileName, 'rb') as f:
	clf_1 = pickle.load(f)

# Load model 2 using pickle
modelFileName = './kfold-models/' + kFoldDirName + '/linear-svm-model-jazz-metal-pop-disco-reggae.pkl'
with open(modelFileName, 'rb') as f:
	clf_2 = pickle.load(f)

# Load datasetTest
testFileName = './kfold-features/' + kFoldDirName + '/test.txt'
with open(testFileName, 'r') as f:
	predictX = f.readlines()
predictX = [x.strip().split(',') for x in predictX]
for p in predictX:
	p.pop()
predictX = np.array(predictX, dtype=float)

# Predict
predictions = []
for test in predictX:
	# Predict for each group
	label_1 = int(clf_1.predict([test]))
	label_2 = int(clf_2.predict([test]))
	label_1 = genres_1[label_1]
	label_2 = genres_2[label_2]

	# Predict using binary classifier
	genres_3 = [label_1, label_2]
	modelFileName = './kfold-models/' + kFoldDirName + '/linear-svm-model-' + label_1 + '-' + label_2 + '.pkl'
	with open(modelFileName, 'rb') as f:
		clf_3 = pickle.load(f)
	label_3 = int(clf_3.predict([test]))
	label_3 = genres_3[label_3]

	# Predict with rock classifier
	genres_4 = ['rock', label_3]
	modelFileName = './kfold-models/' + kFoldDirName + '/linear-svm-model-rock-' + label_3 + '.pkl'
	with open(modelFileName, 'rb') as f:
		clf_4 = pickle.load(f)
	label_4 = int(clf_4.predict([test]))
	label_4 = genres_4[label_4]

	# Add to predictions
	predictions.append(label_4)

print('\n'.join(predictions))
print('Program completed!')