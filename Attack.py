"""
Created On 23rd Sept. 2019

@author: Wushiqiang
"""

import numpy as np
import ffmpeg
import os
import librosa


def noise(x, snr):
    """
    
    """
    
    snr = 10**(snr / 10)
    xpower = np.sum(x ** 2) / x.size
    npower = xpower / snr
    n = np.random.randn(*(x.shape)) * np.sqrt(npower)
    return x + n


def amplitude(x, magnification):
    """
    
    """
    return x * magnification


def cropping(x, duration, start=0, sr=44100):
    """
    """
    idx = start * sr
    rear = (start + duration) * sr
    y = x
    if x.ndim > 1:
        l = x.shape[1]
        if rear > l or start < 0:
            raise Exception("index out of range")
        y[:, 0:idx] = 0
        y[:, rear:] = 0
        return y
    else:
        l = x,shape[0]
        if rear > l or start < 0:
            raise Exception("index out of range")
        y[0: idx] = 0
        y[rear: ] = 0
        return y


def transcode(filepath, dest_format, mono=False):
    """
    输出放在文件夹‘./transcode/’里，由于最后需要load的也是wav格式，所以需要转码回wav格式用于后期读取
    """
    fullFilename = os.path.split(filepath)[-1]
    fname = fullFilename.split('.')[0]
    dest = './transcode/' + fname + '.' + dest_format
    outwav = './transcode/' + fname + dest_format + '.wav'
    (
        ffmpeg
        .input(filepath)
        .output(dest)
        .run()
    )
    (
        ffmpeg
        .input(dest)
        .output(outwav)
        .run()
    )
    y, _ = librosa.load(outwav, sr=44100, mono=mono)
    return y

