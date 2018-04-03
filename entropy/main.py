import numpy as np
import librosa
import glob
import time

def compute_entropy_fd(data, total_sig_energy):
	entropy = np.zeros((data.__len__() - 1024, 1))
	for i in range(data.__len__() - 1024):
		# Get frame of 1024 samples
		frame = data[i:i+1024]

		# Compute fft
		frame_fft = np.fft.fft(frame)

		# Compute power spectrum
		pow = np.abs(np.square(frame_fft))

		# Compute signal energy for frame
		energy = np.sum(pow) / len(pow)

		# Compute proportion
		p = energy / total_sig_energy

		# Compute entropy (energy of the frame divided by the energy of the entire signal)
		if p == 0:
			entropy[i] = 0
		else:
			entropy[i] = p * np.log2(p)

	return entropy

def compute_entropy_td(data, total_sig_energy):
	entropy = np.zeros((data.__len__() - 1024, 1))
	for i in range(data.__len__() - 1024):
		# Get frame of 1024 samples
		frame = data[i:i+1024]

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

# Get current time
ts = time.localtime()
time_str = time.strftime("%Y-%m-%d_%H-%M-%S", ts)

# Open ARFF file
arff_filename = "entropy_" + time_str + ".arff"
arff_file = open(arff_filename, 'w')
arff_file.write("@RELATION music_genre\n@ATTRIBUTE AVG_ENT NUMERIC\n@ATTRIBUTE STD_DEV_ENT NUMERIC\n@ATTRIBUTE MAX_ENT NUMERIC\n@ATTRIBUTE MIN_ENT NUMERIC\n@ATTRIBUTE MAX_DIFF_ENT NUMERIC\n@ATTRIBUTE class {blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock}\n\n@DATA\n")
arff_file.close()

# Genres
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Extract features for each genre
for g in genres:

	print('Processing ' + g + "...")

	# Set directory
	my_path = "train/" + g + "/*.au"

	# Get files
	files = glob.glob(my_path)
	for file in files:

		print(file)

		# Load the example clip
		data, sr = librosa.load(file)
		numSamples = (int) (30.0 * sr)
		data = data[0:numSamples]

		# Compute total signal energy in frequency domain
		#data_fft = np.fft.fft(data)
		#data_pow = np.abs(np.square(data_fft))
		#total_sig_energy_fd = np.sum(data_pow) / len(data_pow)

		# Compute total signal energy in time domain
		data_pow = np.abs(np.square(data))
		total_sig_energy_td = np.sum(data_pow)

		# Compute frequency entropy for each sample
		#p_fd = compute_entropy_fd(data, total_sig_energy_fd)

		# Compute time entropy for each sample
		p_td = compute_entropy_td(data, total_sig_energy_td)

		# Split data into buffers of length 1024 with 50% overlap (or a hop size of 512)
		buffer_size = 1024  # columns
		hop_size = 512
		num_buffers = (int) (np.floor(p_td.__len__() / hop_size) - 1)
		buffer_data = np.zeros((num_buffers, buffer_size))
		entropy_fd = np.zeros((num_buffers, 1))
		entropy_td = np.zeros((num_buffers, 1))
		for i in range(num_buffers):
			# Frequency entropy
			#buf_fd = p_fd[i*hop_size:i*hop_size+buffer_size]
			#ent = - np.sum(buf_fd)
			#entropy_fd[i] = ent

			# Time entropy
			buf_td = p_td[i * hop_size:i * hop_size + buffer_size]
			ent = - np.sum(buf_td)
			entropy_td[i] = ent

		# Compute average entropy of the entropies of each music frame
		#avg_ent_fd = np.mean(entropy_fd)
		avg_ent_td = np.mean(entropy_td)

		# Compute standard deviation of the entropies of each music frame
		#std_dev_ent_fd = np.std(entropy_fd)
		std_dev_ent_td = np.std(entropy_td)

		# Compute maximum entropy among all the entropies of each music frame
		#max_ent_fd = np.max(entropy_fd)
		max_ent_td = np.max(entropy_td)

		# Compute minimum entropy among all the entropies of each music frame
		#min_ent_fd = np.min(entropy_fd)
		min_ent_td = np.min(entropy_td)

		# Compute maximum entropy difference among consecutive frames of the music signal
		#ax_ent_diff_fd = compute_max_ent_diff(entropy_fd)
		max_ent_diff_td = compute_max_ent_diff(entropy_td)

		# Open ARFF file
		arff_file = open(arff_filename, 'a')

		# Write result to ARFF file
		#arff_file.write("{:0.6f},".format(avg_ent_fd))
		#arff_file.write("{:0.6f},".format(std_dev_ent_fd))
		#arff_file.write("{:0.6f},".format(max_ent_fd))
		#arff_file.write("{:0.6f},".format(min_ent_fd))
		#arff_file.write("{:0.6f},".format(max_ent_diff_fd))
		arff_file.write("{:0.6f},".format(avg_ent_td))
		arff_file.write("{:0.6f},".format(std_dev_ent_td))
		arff_file.write("{:0.6f},".format(max_ent_td))
		arff_file.write("{:0.6f},".format(min_ent_td))
		arff_file.write("{:0.6f},".format(max_ent_diff_td))
		arff_file.write("{}\n".format(g))

		# Close file
		arff_file.close()

print('Program completed successfully!\n')