# Beat tracking example
from __future__ import print_function
import librosa
import Utils
import numpy as np

# number of mfcc filters
N_MFCC = 26
# define the complete path of the .mf file listing the songs
PATH_MF = "../test/songs.mf"
# define the path where we can find the two data folders music_wav and speech_wav
PATH_DATA = "test"
# define the complete path of the arff output file
PATH_ARFF_OUT = "./results/TEST_OUT_ARFF.arff"


class Song:
    def __init__(self, s_type="", name=""):
        self._s_type = s_type
        self._name = name
        self._wav = ""
        self._sr = ""
        self.avg_ent_td = ""
        self.std_dev_ent_td = ""
        self.max_ent_td = ""
        self.min_ent_td = ""
        self.max_ent_diff_td = ""
        # [0]mean_mfcc, [1]mean_mfcc_delta, [2]mean_mfcc_delta2, [3]std_mfcc,[4] std_mfcc_delta,[5] std_mfcc_delta2
        # in the current output the mean_mfcc_delta and mean_mfcc_delta2 are not included due to their relevance
        self._mfcc_features = []

    @property
    def s_type(self):
        return self._s_type

    @property
    def name(self):
        return self._name

    @property
    def mfcc_features(self):
        return self._mfcc_features

    @property
    def wav(self):
        return self._wav

    @property
    def sr(self):
        return self._sr

    @name.setter
    def name(self, value):
        self._name = value

    @s_type.setter
    def s_type(self, value):
        self._s_type = value

    @mfcc_features.setter
    def mfcc_features(self, value):
        self._mfcc_features = value

    @wav.setter
    def wav(self, value):
        self._wav = value

    @sr.setter
    def sr(self, value):
        self._sr = value

    # convert the features in string
    def to_string(self):
        s = ""
        s += format(self.avg_ent_td, '.6f') + ","
        s += format(self.std_dev_ent_td, '.6f') + ","
        s += format(self.max_ent_td, '.6f') + ","
        s += format(self.min_ent_td, '.6f') + ","
        s += format(self.max_ent_diff_td, '.6f') + ","
        k = 0;
        for m in self.mfcc_features:
            for i in range(0, N_MFCC):
                # remove the mean delta and mean delta 2
                if (k != 1 and k != 2):
                    s += format(m[i], '.6f') + ","
            k = k + 1;
        s += self.type
        return s


def extract_mfcc_features(song):
    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    # y, sr = librosa.load(filename)
    y = song.wav
    sr = song.sr
    # song creation

    # 3. extract mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # 4. extract mffc delta
    mfcc_delta = librosa.feature.delta(mfcc)

    # 5. extract mfcc delta 2
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # mean
    mean_mfcc = np.mean(mfcc.T, axis=0)
    mean_mfcc_delta = np.mean(mfcc_delta.T, axis=0)
    mean_mfcc_delta2 = np.mean(mfcc_delta2.T, axis=0)

    # std
    std_mfcc = np.std(mfcc.T, axis=0)
    std_mfcc_delta = np.std(mfcc_delta.T, axis=0)
    std_mfcc_delta2 = np.std(mfcc_delta2.T, axis=0)

    song.mfcc_features = (mean_mfcc, mean_mfcc_delta, mean_mfcc_delta2, std_mfcc, std_mfcc_delta, std_mfcc_delta2)


def generate_song_list(path):
    my_songs = []
    lines = Utils.read_file(path)
    for l in lines:
        song = Song()
        # extract filename
        song.name = Utils.get_filename_from_line(l)
        # extract type
        song.type = Utils.get_label_from_line(l)

        my_songs.append(song)
        complete_path = "../" + PATH_DATA + "/" + song.name
        y, sr = librosa.load(complete_path)
        song.wav = y
        song.sr = sr
    return my_songs


songs = generate_song_list(PATH_MF)


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


def extract_entropy(s):
    # Load the example clip
    data = s.wav
    sr = s.sr
    numSamples = (int)(30.0 * sr)
    data = data[0:numSamples]

    # Compute total signal energy in frequency domain
    # data_fft = np.fft.fft(data)
    # data_pow = np.abs(np.square(data_fft))
    # total_sig_energy_fd = np.sum(data_pow) / len(data_pow)

    # Compute total signal energy in time domain
    data_pow = np.abs(np.square(data))
    total_sig_energy_td = np.sum(data_pow)

    # Compute frequency entropy for each sample
    # p_fd = compute_entropy_fd(data, total_sig_energy_fd)

    # Compute time entropy for each sample
    p_td = compute_entropy_td(data, total_sig_energy_td)

    # Split data into buffers of length 1024 with 50% overlap (or a hop size of 512)
    buffer_size = 1024  # columns
    hop_size = 512
    num_buffers = (int)(np.floor(p_td.__len__() / hop_size) - 1)
    buffer_data = np.zeros((num_buffers, buffer_size))
    entropy_fd = np.zeros((num_buffers, 1))
    entropy_td = np.zeros((num_buffers, 1))
    for i in range(num_buffers):
        # Frequency entropy
        # buf_fd = p_fd[i*hop_size:i*hop_size+buffer_size]
        # ent = - np.sum(buf_fd)
        # entropy_fd[i] = ent

        # Time entropy
        buf_td = p_td[i * hop_size:i * hop_size + buffer_size]
        ent = - np.sum(buf_td)
        entropy_td[i] = ent

    # Compute average entropy of the entropies of each music frame
    # avg_ent_fd = np.mean(entropy_fd)
    s.avg_ent_td = np.mean(entropy_td)

    # Compute standard deviation of the entropies of each music frame
    # std_dev_ent_fd = np.std(entropy_fd)
    s.std_dev_ent_td = np.std(entropy_td)

    # Compute maximum entropy among all the entropies of each music frame
    # max_ent_fd = np.max(entropy_fd)
    s.max_ent_td = np.max(entropy_td)

    # Compute minimum entropy among all the entropies of each music frame
    # min_ent_fd = np.min(entropy_fd)
    s.min_ent_td = np.min(entropy_td)

    # Compute maximum entropy difference among consecutive frames of the music signal
    # ax_ent_diff_fd = compute_max_ent_diff(entropy_fd)
    s.max_ent_diff_td = compute_max_ent_diff(entropy_td)


i = 0
print("start")
for s in songs:
    extract_mfcc_features(s)
    extract_entropy(s)
    print(i)
    i = i + 1
Utils.write_arff(songs, PATH_ARFF_OUT)
print("end")
