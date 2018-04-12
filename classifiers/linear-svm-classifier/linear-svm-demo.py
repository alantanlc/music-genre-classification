import sys
import json
import librosa
import numpy as np
import sklearn.svm
import matplotlib.pyplot as py

def main():
	# Get filename from command line argument
	if sys.argv.__len__() != 2:
		print("Example usage: linear-svm-demo.py <au_file_name>")
		return

	# Genres
	genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

	# Load audio
	audioFileName = 'audio-files/' + sys.argv[1]
	# audioFileName = '../../amgc-server/audio-files/' + sys.argv[1]
	data, sr = librosa.load(audioFileName)

	# Generate image of audio plot
	imageName = 'public/images/' + sys.argv[1] + '.png'
	# imageName = '../../amgc-server/public/images/' + sys.argv[1] + '.png'
	py.figure()
	py.plot(data)
	py.title('Time Domain Signal of ' + sys.argv[1])
	py.xlabel('Samples')
	py.ylabel('Amplitude')
	py.savefig(imageName)

	# Extract features
	# ...

	# Load features and labels
	trainFileName = '../features/combined.txt'
	# trainFileName = '../../features/combined.txt'
	with open(trainFileName, 'r') as f:
		X = f.readlines()
	X = [x.strip().split(',') for x in X]
	y = np.array([genres.index(x.pop()) for x in X])
	X = np.array(X, dtype=float)

	# Load feature name
	featuresNameFileName = '../features/features-name.txt'
	# featuresNameFileName = '../../features/features-name.txt'
	with open(featuresNameFileName, 'r') as f:
		N = f.readlines()
	N = [n.strip() for n in N]

	# Train classifier
	clf = sklearn.svm.SVC(kernel='linear', C=1.0)
	clf.fit(X, y)

	# Predict
	test = X[0]
	label = int(clf.predict([test]))

	# Return results to node server
	jsonResult = {
		'filename': sys.argv[1],
		'image': sys.argv[1] + '.png',
		'label': genres[label],
		'features': N,
		'values': list(test),
		'confidence': 80
	}
	print(json.dumps(jsonResult))

if __name__ == "__main__":
	main()