import sys
import json
import pickle
import librosa
import numpy as np
import sklearn.svm
import matplotlib.pyplot as py
import extractor

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
	test = extractor.get_features(data, sr)

	# Load feature name
	featuresNameFileName = '../features/features-name.txt'
	with open(featuresNameFileName, 'r') as f:
		N = f.readlines()
	N = [n.strip() for n in N]

	# Load model using pickle
	modelFileName = 'linear-svm-model.pkl'
	with open(modelFileName, 'rb') as f:
		clf = pickle.load(f)

	# Predict
	label = int(clf.predict([test]))

	# Return results to node server
	jsonResult = {
		'filename': sys.argv[1],
		'image': sys.argv[1] + '.png',
		'label': genres[label],
		'features': N,
		'values': list(test)
	}
	print(json.dumps(jsonResult))

if __name__ == "__main__":
	main()