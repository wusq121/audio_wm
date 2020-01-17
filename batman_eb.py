"""
Creat on 17th Sept. 2019

@author: Wushiqiang
"""

import transform as tf
import Preprocess as pre
import Embedding as eb
import librosa
import numpy as np
import os

def embed(y, n_dt, code, P, n_code=32):
    """
    """
    y = y.flatten()
    Dc = tf.stt_my(y, n_dt=n_dt)
    if code == 'random':
        wmbits = np.random.choice(16, size=n_code)
    else:
        wmbits = pre.hex_to_code(code)
    ns = pre.code_to_hex(wmbits)
    
    signal_wmd = np.zeros_like(Dc)
    wt = pre.time_weight(Dc, N=n_code, L_n=P.shape[1])
    for i in range(Dc.shape[1]):
        X_i = Dc[:, i]
        X_l, X_selected, X_h = pre.dct_segment_generate(X_i, N=n_code, L_n=P.shape[1])
        Y = eb.watermark_embed(X_selected, P, wmbits, N=n_code, weight=wt[i])
        Y_i = pre.dct_reconstruct(X_l, Y, X_h)
        signal_wmd[:, i] = Y_i
    embeded = tf.istt_my(signal_wmd, length=y.shape[0])
    
    return embeded, ns


path = 'F:/audio_wm/audio/'
all_file = pre.filter_file(path, 'wav')
#all_file = ['./audio/batman-5min.wav']

n_dt = 8192
L_n = 32
n_code = 32 # 比特数除以4， 这里比特数是128


p0 = eb.seed_generate(L_n)
P = eb.pn_code_generate(16, p0)
np.save('F:/audio_wm/data/p0.npy', p0)
for filepath in all_file:
    print("Embedding in " + filepath)
    aFullFilename = os.path.split(filepath)[-1]
    filename = aFullFilename.split('.')[0]
    audio, sr = librosa.load(filepath, sr=44100, mono=False)
    
    embedded = np.zeros_like(audio)
    ns = '841F-4483-3ABE-A5D0-E496-5B23-CFA1-2AD7'
    if audio.ndim > 1:
        for j in range(audio.shape[0]):
            y = audio[j, :]
            embedded[j, :], ns = embed(y, n_dt, ns, P, n_code=32)
    else:
        embedded, ns = embed(audio, n_dt, ns, P, n_code=32)
    
    
    with open('F:/audio_wm/data/'+ filename +'.txt', 'a') as fp:
        fp.write('嵌入的码字为: '+ '\n' + ns + '\n')
    
    librosa.output.write_wav('F:/audio_wm/result/'+filename + '.wav', embedded, sr)
    print(filename + 'watermark signal ratio(dB): ', eb.snr(audio, embedded))