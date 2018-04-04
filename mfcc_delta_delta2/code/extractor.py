# Beat tracking example
from __future__ import print_function
import librosa
import Utils
import numpy as np

# number of mfcc filters
N_MFCC = 26
# define the complete path of the .mf file listing the songs
PATH_MF = "../train/songs.mf"
# define the path where we can find the two data folders music_wav and speech_wav
PATH_DATA = "train"
# define the complete path of the arff output file
PATH_ARFF_OUT = "./results/OUT_ARFF.arff"


class Song:
    def __init__(self, s_type="", name=""):
        self._s_type = s_type
        self._name = name
        self._wav = ""
        self._sr = ""
        # mean_mfcc, mean_mfcc_delta, mean_mfcc_delta2, std_mfcc, std_mfcc_delta, std_mfcc_delta2
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
        for m in self.mfcc_features:
            for i in range(0,N_MFCC):
                  s += format(m[i], '.6f') + ","
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
        complete_path = "../"+PATH_DATA + "/" + song.name
        y, sr = librosa.load(complete_path)
        song.wav = y
        song.sr = sr
    return my_songs


songs = generate_song_list(PATH_MF)
for s in songs:
    extract_mfcc_features(s)
Utils.write_arff(songs, PATH_ARFF_OUT)
print("end")
