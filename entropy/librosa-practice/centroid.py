import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Get the file path to the included audio example
filename = 'Train/blues/blues.00000.au'

# Load the example clip
y, sr = librosa.load(filename)

# Compute spectral centroid
sc = librosa.feature.spectral_centroid(y=y, sr=sr)

# Compute spectrogram
S, phase = librosa.magphase(librosa.stft(y=y))
librosa.feature.spectral_centroid(S=S)

# Plot the result
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(sc.T, label='Spectral centroid')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, sc.shape[-1]])
plt.legend()
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()