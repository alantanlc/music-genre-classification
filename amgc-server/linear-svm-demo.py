import sys
import json
import pickle
import librosa
import extractor
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
	timeImageName = 'public/images/' + sys.argv[1] + '-time.png'
	py.figure()
	py.plot(data)
	py.title('Time Domain Signal')
	py.xlabel('Samples')
	py.ylabel('Amplitude')
	py.savefig(timeImageName)

	# Extract features
	test = extractor.get_features(data, sr)

	# Generate image of audio entropy features
	entropyImageName = 'public/images/' + sys.argv[1] + '-entropy.png'
	py.figure()
	py.plot(test[0:5])
	py.title('Entropy Features')
	py.xlabel('Features')
	py.ylabel('Values')
	py.savefig(entropyImageName)

	# Generate image of mfcc features
	mfccImageName = 'public/images/' + sys.argv[1] + '-mfcc.png'
	py.figure()
	py.plot(test[5:])
	py.title('MFCC Features')
	py.xlabel('Features')
	py.ylabel('Values')
	py.savefig(mfccImageName)

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

	# Get wiki content
	wikiFileName = "wiki.json"
	with open(wikiFileName) as json_data:
		wiki = json.load(json_data)

	# Return results to node server
	jsonResult = {
		'filename': sys.argv[1],
		'time_image': sys.argv[1] + '-time.png',
		'entropy_image': sys.argv[1] + '-entropy.png',
		'mfcc_image': sys.argv[1] + '-mfcc.png',
		'label': genres[label],
		'features': N,
		'values': list(test),
		'wiki': wiki[genres[label]]
	}
	print(json.dumps(jsonResult))

if __name__ == "__main__":
	main()