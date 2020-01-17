"""
Created on 10th Sept. 2019

@author: Wushiqiang
"""

import librosa
import numpy as np
import transform
import os
import fnmatch

def dct_segment_generate(X, N=32, L_n=32):
    """
    由于每帧的变换对应的dct变换不会太长，所以用20个16维的序列嵌入80bit的信息
    :param X: 某一帧对应的DCT信号
    :param N: 需要嵌入的码字数量，默认16个
    :param L_n: 需要嵌入的随机码字的长度，这里默认为16，这样可以嵌入的数量为4bits
    :return X_pre: 无需嵌入的低频部分
    :return X_selected: 嵌入码字的中频部分
    :return X_suffix: 无需嵌入的高频部分
    """
    
    length = X.shape[0]
    
    idx = int(length // 8)
    
    X_pre = X[0:idx]
    X_selected = np.zeros([N, L_n], dtype=X.dtype)
    for i in range(N):
        X_selected[i] = X[idx:(idx + L_n)]
        idx = idx + L_n
    X_suffix = X[idx:]
    return X_pre, X_selected, X_suffix

def dct_reconstruct(X_pre, X_selected, X_suffix):
    """
    将分帧的信号重新组织为一维信号
    :param X_pre: 低频信号
    :param X_selected: 中频信号
    :param X_suffix:高频信号
    :return X:原始频域信号
    """
    N, L_n = X_selected.shape
    idx = X_pre.shape[0]
    L = idx + N * L_n + X_suffix.shape[0]
    
    X_n = np.zeros(L, dtype=X_pre.dtype)
    X_n[0:idx] = X_pre
    for i in range(N):
        X_n[idx:(idx + L_n)] = X_selected[i, :]
        idx += L_n
    X_n[idx: ] = X_suffix
    
    return X_n
    

def code_to_hex(wmbits):
    """
    将嵌入的码字得到16进制的字符串用于显式表达
    :param wmbits：码字
    :return ns: 16进制显式字符串
    """
    dic = '0123456789ABCDEF'
    y = [
        dic[j] for j in wmbits
    ]
    n = len(y)
    done = 0
    for i in range(n):
        if (i % 4 == 3):
            y.insert(i+done+1, '-')
            done += 1
    
    y.pop()
    ns = ''
    ns = ns.join(y)
    return ns


def hex_to_code(ns):
    """
    将字符串化的16进制码字转换为数字
    :param ns: 16进制的显式zifuc
    :return wmbits: 码字
    """
    ns = ns.replace('-', '')
    dict = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'A': 10,
        'B': 11,
        'C': 12,
        'D': 13,
        'E': 14,
        'F': 15
    }
    wmbits = []
    for i in range(len(ns)):
        wmbits.append(dict[ns[i]])
    return np.array(wmbits, dtype=np.int_)


def time_weight(Dc, N=20, L_n=32, mode='square'):
    """
    计算时序上的权重
    :param Dc: 音频信号的短时DCT变换矩阵
    :param N: 嵌入码字的数量
    :param L_n: 单个码字的长度
    :param mode: ['square', 'sum']
    :return weight: 向量，每帧的强度权重。
    """
    
    idx = int(Dc.shape[0] // 8)
    X_selected = Dc[idx:(idx + N * L_n), :]
    
    ss = np.zeros(Dc.shape[1])
    for i in range(Dc.shape[1]):
        x_i = X_selected[:, i].flatten()
        if mode == 'square':
            ss[i] = np.sum(x_i ** 2)
        elif mode == 'sum':
            ss[i] = np.sum(x_i)
    
    weight = ss / np.max(ss)
    ws = np.sort(weight)
    nidx = [Dc.shape[1]//4, Dc.shape[1] //2, (Dc.shape[1] * 3) // 4]
    bins = np.array([0, ws[nidx[0]], ws[nidx[1]], ws[nidx[2]], 1])
    indices = np.digitize(weight, bins)
    space = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    for i in range(Dc.shape[1]):
        weight[i] = space[indices[i]]
    return weight
    
    
def filter_file(path, file_ext):
    flist = os.listdir(path)
    all_file = []
    for filename in flist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            all_file.extend(filter_file(filepath, file_ext))
        elif fnmatch.fnmatch(filepath, '*.' + file_ext):
            all_file.append(filepath)
        else:
            pass
    
    return all_file


def seed_generate(L_n):
    tmp = np.random.randn(L_n)
    return np.sign(tmp)


def pn_sequences(n, p0):
    """
    生成n个长度为L_n的随机码字，这些码字相互正交，模为1。
    为防止出现循环矩阵不是满秩矩阵的情况，这里有n > L_n
    :param n: 随机码字的个数
    :param p0: 随机码字的种子，用于再现随机码
    :return P: 生成的随机码字组成的矩阵
    """
    L_n = p0.shape[0]
    mat = np.zeros([L_n, L_n])
    
    for i in range(L_n):
        mat[i, :] = np.roll(p0, i)
    
    P = np.zeros([L_n, L_n])
    for i in range(L_n):
        if i == 0:
            P[i, :] = mat[i, :]
            P[i, :] = P[i, :] / np.linalg.norm(P[i, :])
        else:
            tmp = np.zeros(L_n)
            for j in range(i):
                tmp += np.dot(mat[i, :], P[j, :]) * P[j, :]
            P[i, :] = mat[i, :] - tmp
            P[i, :] = P[i, :] / np.linalg.norm(P[i, :])
    return P[: n, :]