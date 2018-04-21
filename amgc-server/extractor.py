import sys
import librosa
import numpy as np


def extract_mfcc_features(y, sr):
	# Number of mfcc filters
	N_MFCC = 26

	# Extract mfcc
	mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

	# Extract mfcc delta
	mfcc_delta = librosa.feature.delta(mfcc)

	# Extract mfcc delta 2
	mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

	# Mean
	mean_mfcc = np.mean(mfcc.T, axis=0).tolist()
	# mean_mfcc_delta = np.mean(mfcc_delta.T, axis=0).tolist()
	# mean_mfcc_delta2 = np.mean(mfcc_delta2.T, axis=0).tolist()

	# Std
	std_mfcc = np.std(mfcc.T, axis=0).tolist()
	std_mfcc_delta = np.std(mfcc_delta.T, axis=0).tolist()
	std_mfcc_delta2 = np.std(mfcc_delta2.T, axis=0).tolist()

	mfcc_features = mean_mfcc + std_mfcc + std_mfcc_delta + std_mfcc_delta2
	return mfcc_features


def compute_entropy_td(data, total_sig_energy):
	entropy = np.zeros((data.__len__() - 1024, 1))
	for i in range(data.__len__() - 1024):
		# Get frame of 1024 samples
		frame = data[i:i + 1024]

		# Compute power spectrum
		pow = np.abs(np.square(frame))

		# Compute signal energy for frame
		energy = np.sum(pow)

		# Compute proportion
		p = energy / total_sig_energy

		# Compute entropy (energy of the frame divided by the energy of the entire signal)
		if p == 0:
			entropy[i] = 0
		else:
			entropy[i] = p * np.log2(p)

	return entropy


def compute_max_ent_diff(data):
	diff = data[1:] - data[0:-1]
	return np.max(diff)


def extract_entropy(y, sr):
	# Truncate song if required
	numSamples = (int)(30.0 * sr)
	data = y[0:numSamples]

	# Compute total signal energy in time domain
	data_pow = np.abs(np.square(data))
	total_sig_energy_td = np.sum(data_pow)

	# Compute time entropy for each sample
	p_td = compute_entropy_td(data, total_sig_energy_td)

	# Split data into buffers of length 1024 with 50% overlap (or a hop size of 512)
	buffer_size = 1024  # columns
	hop_size = 512
	num_buffers = (int)(np.floor(p_td.__len__() / hop_size) - 1)
	entropy_td = np.zeros((num_buffers, 1))
	for i in range(num_buffers):
		# Time entropy
		buf_td = p_td[i * hop_size:i * hop_size + buffer_size]
		ent = - np.sum(buf_td)
		entropy_td[i] = ent

	# Compute average entropy of the entropies of each music frame
	avg_ent_td = np.mean(entropy_td)

	# Compute standard deviation of the entropies of each music frame
	std_dev_ent_td = np.std(entropy_td)

	# Compute maximum entropy among all the entropies of each music frame
	max_ent_td = np.max(entropy_td)

	# Compute minimum entropy among all the entropies of each music frame
	min_ent_td = np.min(entropy_td)

	# Compute maximum entropy difference among consecutive frames of the music signal
	max_ent_diff_td = compute_max_ent_diff(entropy_td)

	return [avg_ent_td, std_dev_ent_td, max_ent_td, min_ent_td, max_ent_diff_td]

def get_features(y, sr):
	features = []

	# Extract entropy
	features = features + extract_entropy(y, sr)

	# Extract mfcc
	features = features + extract_mfcc_features(y, sr)

	# Format result
	features = [float(format(x, '.6f')) for x in features]

	# Return features
	return features

def main():
	# Get audio file name through command line argument
	filename = sys.argv[1]

	# Load audio using librosa
	y, sr = librosa.load(filename)

	# Extract entropy
	features = get_features(y, sr)

	# Print
	print(features)

if __name__ == "__main__":
	main()
