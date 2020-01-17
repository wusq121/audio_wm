"""
Created on 26th Aug. 2019

@author: Wushiqiang
"""

import librosa
import scipy.fftpack as fft
import numpy as np

def stft_my(x, n_fft=2048, hop_length=None, win_length=None, window='hann',
            center=True, dtype=np.complex64, pad_mode='reflect'):
    if win_length is None:
        win_length = n_fft
    
    if hop_length is None:
        hop_length = int(win_length // 4)
    
    fft_window = librosa.filters.get_window(window, win_length, fftbins=True)
    
    fft_window = librosa.util.pad_center(fft_window, n_fft)
    
    fft_window = fft_window.reshape((-1, 1))
    
    librosa.util.valid_audio(y)
    
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)
        
    y_frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)
    
    stft_mat = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                        dtype=dtype,
                        order='F')
    
    n_columns = int(librosa.util.MAX_MEM_BLOCK / (stft_mat.shape[0] * stft_mat.itemsize))
    
    for bl_s in range(0, stft_mat.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_mat.shape[1])
        
        stft_mat[:, bl_s:bl_t] = fft.fft(fft_window * y_frames[:, bl_s:bl_t], axis=0)[:stft_mat.shape[0]]
    
    return stft_mat


def istft_my(stft_mat, hop_length=None, win_length=None, window='hann', 
             center=True, dtype=np.float32, length=None):
    
    n_fft = 2 * (stft_mat.shape[0] - 1)
    
    if win_length is None:
        win_length = n_fft
    
    if hop_length is None:
        hop_length = int(win_length // 4)
        
    ifft_window = librosa.filters.get_window(window, win_length, fftbins=True)
    
    ifft_window = librosa.util.pad_center(ifft_window, n_fft)
    
    n_frames = stft_mat.shape[1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_mat[:, i].flatten()
        spec = np.concatenate((spec, spec[-2:0:-1].conj()), 0)
        ytmp = ifft_window * fft.ifft(spec).real
        
        y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp
    
    ifft_window_sum = librosa.filters.window_sumsquare(window, 
                                                       n_frames, 
                                                       win_length=win_length,
                                                       n_fft=n_fft,
                                                       dtype=dtype)
    
    approx_nonzero_idx = ifft_window_sum > librosa.util.tiny(ifft_window_sum)
    y[approx_nonzero_idx] /= ifft_window_sum[approx_nonzero_idx]
    
    if length is None:
        if center:
            y = y[int(n_fft // 2): -int(n_fft // 2)]
    
    else:
        if center:
            start = int(n_fft // 2)
        else:
            start = 0
        
        y = librosa.util.fix_length(y[start:], length)
    
    return y


def stt_my(x, n_dt=2048, hop_length=None, win_length=None, window='hann', 
            center=True, dtype=np.float_, pad_mode='reflect', base='c'):
    
    if base == 'c':
        trans = fft.dct
    elif base == 's':
        trans = fft.dst
    
    if win_length is None:
        win_length = n_dt
    
    if hop_length is None:
        hop_length = int(win_length // 4)
    
    dct_window = librosa.filters.get_window(window, win_length, fftbins=True)
    
    dct_window = librosa.util.pad_center(dct_window, n_dt)
    
    dct_window = dct_window.reshape((-1, 1))
    
    librosa.util.valid_audio(x)
    
    if center:
        x = np.pad(x, int(n_dt // 2), mode=pad_mode)
    
    x_frames = librosa.util.frame(x, frame_length=n_dt, hop_length=hop_length)
    
    stt_mat = np.empty((n_dt, x_frames.shape[1]), 
                        dtype=dtype,
                        order='F')
    
    n_columns = int(librosa.util.MAX_MEM_BLOCK / (stt_mat.shape[0] * stt_mat.itemsize))
    
    for bl_s in range(0, stt_mat.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stt_mat.shape[1])
        
        stt_mat[:, bl_s:bl_t] = trans(dct_window * x_frames[:, bl_s:bl_t], type=2, norm='ortho', axis=0)
    
    return stt_mat


def istt_my(stt_mat, hop_length=None, win_length=None, window='hann',
             center=True, dtype=np.float32, length=None, base='c'):
    
    if base == 'c':
        trans = fft.idct
    elif base == 's':
        trans = fft.idst
    
    n_dt = stt_mat.shape[0]
    
    if win_length is None:
        win_length = n_dt
    
    if hop_length is None:
        hop_length = int(win_length // 4)
    
    idt_window = librosa.filters.get_window(window, win_length, fftbins=True)
    
    idt_window = librosa.util.pad_center(idt_window, n_dt)
    
    n_frames = stt_mat.shape[1]
    expected_signal_len = n_dt + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    
    for i in range(n_frames):
        sample = i * hop_length
        spec = stt_mat[:, i].flatten()
        #spec = np.concatenate((spec, spec[-2:0:-1].conj()), 0)
        ytmp = idt_window * trans(spec, type=2, norm='ortho')
        
        y[sample:(sample + n_dt)] = y[sample:(sample + n_dt)] + ytmp
    
    idt_window_sum = librosa.filters.window_sumsquare(window, 
                                      n_frames, 
                                      win_length=win_length, 
                                      n_fft=n_dt, 
                                      hop_length=hop_length, 
                                      dtype=dtype)
    
    approx_nonzero_idx = idt_window_sum > librosa.util.tiny(idt_window_sum)
    y[approx_nonzero_idx] /= idt_window_sum[approx_nonzero_idx]
    
    if length is None:
        
        if center:
            y = y[int(n_dt // 2): -int(n_dt // 2)]
    else:
        if center:
            start = int(n_dt // 2)
        else:
            start = 0
        y = librosa.util.fix_length(y[start:], length)
    
    return y
    