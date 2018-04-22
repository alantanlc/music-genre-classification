import pickle
import numpy as np

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
genres_1 = ['blues','classical','country','hiphop']
genres_2 = ['jazz','metal','pop','disco','reggae']

# Load model 1 using pickle
modelFileName = 'models/linear-svm-model-blues-classical-country-hiphop.pkl'
with open(modelFileName, 'rb') as f:
	clf_1 = pickle.load(f)

# Load model 2 using pickle
modelFileName = 'models/linear-svm-model-jazz-metal-pop-disco-reggae.pkl'
with open(modelFileName, 'rb') as f:
	clf_2 = pickle.load(f)

# Load datasetTest
testFileName = './features/predict-features.txt'
with open(testFileName, 'r') as f:
	predictX = f.readlines()
predictX = [x.strip().split(',') for x in predictX]
predictX = np.array(predictX, dtype=float)

# Predict
predictions = []
for test in predictX:
	# Predict for Genres 1
	scalerFileName = './scaler/scaler-blues-classical-country-hiphop.pkl'
	with open(scalerFileName, 'rb') as f:
		scaler = pickle.load(f)
	test_reshape = test.reshape(1,-1)
	test_scaler = scaler.transform(test_reshape)
	label_1 = int(clf_1.predict(test_scaler))
	label_1 = genres_1[label_1]

	# Predict for Genres 2
	scalerFileName = './scaler/scaler-jazz-metal-pop-disco-reggae.pkl'
	with open(scalerFileName, 'rb') as f:
		scaler = pickle.load(f)
	test_reshape = test.reshape(1, -1)
	test_scaler = scaler.transform(test_reshape)
	label_2 = int(clf_2.predict(test_scaler))
	label_2 = genres_2[label_2]

	# Predict using binary classifier
	genres_3 = [label_1, label_2]
	modelFileName = 'models/linear-svm-model-' + label_1 + '-' + label_2 + '.pkl'
	with open(modelFileName, 'rb') as f:
		clf_3 = pickle.load(f)
	scalerFileName = './scaler/scaler-' + label_1 + '-' + label_2 + '.pkl'
	with open(scalerFileName, 'rb') as f:
		scaler = pickle.load(f)
	test_reshape = test.reshape(1, -1)
	test_scaler = scaler.transform(test_reshape)
	label_3 = int(clf_3.predict(test_scaler))
	label_3 = genres_3[label_3]

	# Predict with rock classifier
	genres_4 = ['rock', label_3]
	modelFileName = 'models/linear-svm-model-rock-' + label_3 + '.pkl'
	with open(modelFileName, 'rb') as f:
		clf_4 = pickle.load(f)
	scalerFileName = './scaler/scaler-rock-' + label_3 + '.pkl'
	with open(scalerFileName, 'rb') as f:
		scaler = pickle.load(f)
	test_reshape = test.reshape(1, -1)
	test_scaler = scaler.transform(test_reshape)
	label_4 = int(clf_4.predict(test_scaler))
	label_4 = genres_4[label_4]

	# Add to predictions
	predictions.append(label_4)

print('\n'.join(predictions))
print('Program completed!')