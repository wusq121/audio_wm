"""
Created on 19th Sept. 2019

@author: Wushiqiang
"""

import numpy as np
import transform as tf
import Extraction as et
import Preprocess as pre
import os
import librosa
import Attack as atk


def extract_vote(y, n_dt, P, n_code):
    """
    """
    wmbits = []
    y = y.flatten()
    Dc = tf.stt_my(y, n_dt=n_dt)
    for i in range(Dc.shape[1]):
        X_i = Dc[:, i]
        X_l, X_s, X_h = pre.dct_segment_generate(X_i, N=n_code, L_n=P.shape[1])
        wmt = et.extract(X_s, P)
        if wmt is not None:
            wmbits.append(wmt)
    return wmbits


def callAttack(filepath, attack, **kargs):
    """
    """
    mono = kargs.get('mono', True)
    x, sr = librosa.load(filepath, sr=44100, mono=False)

    if attack == 'transcode':
        dest_format = kargs.get('dest_format', 'mp3')
        y = atk.transcode(filepath, dest_format, mono)
        status = 'transcode' + dest_format
    elif attack == 'noise':
        snr = kargs.get('snr', 10)
        y = atk.noise(x, snr)
        status = 'noise' + str(snr) + 'dB'
    elif attack == 'cropping':
        duration = kargs.get('duration', 5)
        start = kargs.get('start', 0)
        sr = kargs.get('sr', 44100)
        y = atk.cropping(x, duration, start, sr)
        status = 'cropping-' + 'start-' + str(start) + '-duration-' + str(duration)
    elif attack == 'amplitude':
        magnification = kargs.get('magnification', 2)
        y = atk.amplitude(x, magnification)
        status = 'amplitude' + str(magnification)
    else:
        y = x
        status = 'None'
    return y, sr, status
        


path = './result/'
#all_file = pre.filter_file(path, 'wav')
all_file = ['./result/batman-5min.wav']
#attack_bank =['None', 'noise', 'cropping', 'transcode', 'amplitude']
attack_bank = ['None']
kargs = [{'dest_format': 'mp3', 'snr': 15, 'duration': 3, 'start': 150, 'magnification': 1.7, 'mono': False}, 
         {'dest_format': 'wma', 'snr': 5, 'duration': 7, 'start': 75, 'magnification': 1.3, 'mono': False}]

for filepath in all_file:
    aFilename = os.path.split(filepath)[-1]
    filename = aFilename.split('.')[0]
    
    n_dt = 8192
    n_code = 32
    p0 = np.load('./data/' + filename + '.npy')
    P = et.pn_reconstruct(16, p0)
    for attack in attack_bank:
        print(attack)
        for key in kargs:
            print(key)
            wm = []
            audio, sr, status = callAttack(filepath, attack, **key)
            if audio.ndim > 1:
                for j in range(audio.shape[0]):
                    y = audio[j, :]
                    wm.extend(extract_vote(y, n_dt, P, n_code))
            else:
                wm.extend(extract_vote(y, n_dt, P, n_code))
            
            wm = et.vote(np.asarray(wm, dtype=np.int_))
            ns = pre.code_to_hex(wm)
            
            with open('./data/' + filename + '.txt', 'a') as fp:
                fp.write(status + ', 提取出来的码字: ' + '\n' + ns + '\n')
