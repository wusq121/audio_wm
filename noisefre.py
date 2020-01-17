"""
Created on Oct. 14th 2019

@author: wushiqiang
"""

import librosa
import Attack as at
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


filename = './embedded/batman-5min8192.wav'
x, sr = librosa.load(filename, sr=44100, mono=False)
y = at.noise(x, 5)
librosa.output.write_wav('5dB.wav', y.astype(np.float32), sr, norm=True)
plt.figure()
plt.subplot(2,1,1)
D1 = np.abs(librosa.stft(y[0, :]))
librosa.display.specshow(librosa.amplitude_to_db(D1, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

plt.subplot(2, 1, 2)
D2 = np.abs(librosa.stft(y[1, :]))
librosa.display.specshow(librosa.amplitude_to_db(D2, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

plt.figure()
plt.subplot(2,1,1)
D3 = np.abs(librosa.stft(x[0, :]))
librosa.display.specshow(librosa.amplitude_to_db(D3, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

plt.subplot(2, 1, 2)
D4 = np.abs(librosa.stft(x[1, :]))
librosa.display.specshow(librosa.amplitude_to_db(D4, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()