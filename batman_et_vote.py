"""
Creat on 18th Sept. 2019

@author: Wushiqiang
"""
import librosa
import Preprocess as pre
import Extraction as et
import transform as tf
import numpy as np
import os

def extract(y, n_dt, P, wmbits, n_code):
    """
    """
    Dc = tf.stt_my(y, n_dt=n_dt)
    for i in range(Dc.shape[1]):
        X_i = Dc[:, i]
        X_l, X_s, X_h = pre.dct_segment_generate(X_i, N=n_code, L_n=P.shape[1])
        wmt = et.extract(X_s, P)
        wmbits.append(wmt)
    return wmbits

        
path = 'F:/audio_wm/result/'
all_file = pre.filter_file(path, 'wav')
p0 = np.load('F:/audio_wm/data/p0.npy')
P = et.pn_reconstruct(16, p0)

for filepath in all_file:
    aFilename = os.path.split(filepath)[-1]
    filename = aFilename.split('.')[0]
    
    
    audio, sr = librosa.load(filepath, sr=44100, mono=False)
    
    n_dt = 8192
    n_code = 32
    #p0 = np.load('./data/' + filename +'.npy')
    
    #P = et.pn_reconstruct(16, p0)
    wmbits = []
    if audio.ndim > 1:
        for j in range(audio.shape[0]):
            y = audio[j, :]
            wmbits = extract(y, n_dt, P, wmbits, n_code)
    else:
        wmbits = extract(audio, n_dt, P, wmbits, n_code)
    
    wm = et.vote(np.asarray(wmbits, dtype=np.int_))
    ns = pre.code_to_hex(wm)
    
    with open('F:/audio_wm/data/' + filename + '.txt', 'a') as fp:
        fp.write(path + '中提取出来的码字: ' + '\n' + ns + '\n')