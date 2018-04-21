import numpy as np
import librosa
import glob
import time


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
	# Data
	data = y

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


def extract_beat_features(y, sr):
	# Extract tempo
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

	# Beat frames
	beat_frame_diff = beat_frames[1:] - beat_frames[:-1]
	avg_beat_frame_diff = np.mean(beat_frame_diff)

	# Zero crossing
	y_zcr = np.mean(librosa.feature.zero_crossing_rate(y))

	# Harmonic
	y_harmonic = np.mean(librosa.effects.harmonic(y))

	# Percussive
	y_percussive = np.mean(librosa.effects.percussive(y))

	# Features
	features = [tempo, avg_beat_frame_diff, y_zcr, y_harmonic, y_percussive]

	return features

def get_features(y, sr):
	# Features
	features = []

	# Extract entropy
	features = features + extract_entropy(y, sr)

	# Extract mfcc
	features = features + extract_mfcc_features(y, sr)

	# Extract beat
	features = features + extract_beat_features(y, sr)

	# Format result
	features = [float(format(x, '.6f')) for x in features]

	# Return features
	return features

# Get current time
ts = time.localtime()
time_str = time.strftime("%Y-%m-%d_%H-%M-%S", ts)

# Open ARFF file
arff_filename = "beat-features-" + time_str + ".arff"
arff_file = open(arff_filename, 'w')
arff_file.write("@RELATION music_genre\n"
                "@ATTRIBUTE AVG_ENT NUMERIC\n"
                "@ATTRIBUTE STD_DEV_ENT NUMERIC\n"
                "@ATTRIBUTE MAX_ENT NUMERIC\n"
                "@ATTRIBUTE MIN_ENT NUMERIC\n"
                "@ATTRIBUTE MAX_DIFF_ENT NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_0 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_1 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_2 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_3 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_4 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_5 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_6 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_7 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_8 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_9 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_10 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_11 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_12 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_13 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_14 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_15 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_16 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_17 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_18 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_19 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_20 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_21 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_22 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_23 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_24 NUMERIC\n"
				"@ATTRIBUTE MEAN_MFCC_25 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_0 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_1 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_2 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_3 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_4 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_5 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_6 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_7 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_8 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_9 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_10 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_11 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_12 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_13 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_14 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_15 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_16 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_17 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_18 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_19 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_20 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_21 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_22 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_23 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_24 NUMERIC\n"
				"@ATTRIBUTE STD_MFCC_25 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_0 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_1 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_2 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_3 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_4 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_5 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_6 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_7 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_8 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_9 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_10 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_11 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_12 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_13 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_14 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_15 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_16 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_17 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_18 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_19 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_20 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_21 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_22 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_23 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_24 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA_MFCC_25 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_0 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_1 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_2 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_3 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_4 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_5 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_6 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_7 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_8 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_9 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_10 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_11 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_12 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_13 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_14 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_15 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_16 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_17 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_18 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_19 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_20 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_21 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_22 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_23 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_24 NUMERIC\n"
				"@ATTRIBUTE STD_DELTA2_MFCC_25 NUMERIC\n"
				"@ATTRIBUTE TEMPO NUMERIC\n"
				"@ATTRIBUTE AVG_BEAT_FRAME_DIFF NUMERIC\n"
				"@ATTRIBUTE AVG_ZCR NUMERIC\n"
				"@ATTRIBUTE AVG_HARMONIC NUMERIC\n"
				"@ATTRIBUTE AVG_PERCUSSIVE NUMERIC\n"
                "@ATTRIBUTE class {blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock}\n\n"
                "@DATA\n")
arff_file.close()

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Print file name
print(arff_filename)

# Extract features for each genre
for g in genres:

	print('Processing ' + g + "...")

	# Set directory
	my_path = "D:/PyCharmProjects/train/" + g + "/*.au"

	# Get files
	files = glob.glob(my_path)
	for file in files:
		# Log file name to show progress
		print(file)

		# Load the example clip
		duration_seconds = 10.0
		y, sr = librosa.load(file)
		numSamples = (int) (duration_seconds * sr)
		y = y[0:numSamples]

		# Features
		features = get_features(y, sr)

		# Convert from float to string
		features = [str(x) for x in features]

		# Open ARFF file
		arff_file = open(arff_filename, 'a')

		# Write result to ARFF file
		arff_file.write(",".join(features))
		arff_file.writelines(",")
		arff_file.write("{}\n".format(g))

		# Close file
		arff_file.close()

print('Program completed successfully!\n')