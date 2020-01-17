"""
Created on 11th Sept. 2019

@author: Wushiqiang
"""

import numpy as np


def extract(Y, p0):
    """
    将水印信息提取出来
    :param Y: 包含水印信息的频域信息
    :param p0: 伪随机码组成的种子信息
    :return wmbits: 嵌入的水印信息
    """
    N = Y.shape[0]
    wmbits = np.zeros(N, dtype=np.int_)
    P = pn_reconstruct(p0)
    
    for j in range(N):
        tmp = np.abs(np.dot(Y[j, :], P.T))
        wmbits[j] = tmp.argmax()
    return wmbits


def pn_reconstruct(p0):
    """
    根据保存的种子信息，重建伪随机码字矩阵
    :param p0: 种子序列
    :return P: 伪随机码字矩阵
    """
    L_n = p0.shape[0]
    mat = np.zeros([L_n, L_n], dtype=np.float32)
    for i in range(L_n):
        mat[i, :] = np.roll(p0, i)
    
    P = np.zeros_like(mat, dtype=np.float32)
    for i in range(L_n):
        if i == 0:
            P[i, :] = mat[i, :]
            P[i, :] = P[i, :] / np.linalg.norm(P[i, :])
        else:
            tmp = np.zeros(L_n, dtype=np.float32)
            for j in range(i):
                tmp += np.dot(mat[i, :], P[j, :]) * P[j, :]
            P[i, :] = mat[i, :] - tmp
            P[i, :] = P[i, :] / np.linalg.norm(P[i, :])
    return 10 * P


def vote(wmbits):
    """
    输入是n_frames * N的矩阵，输出是N为数组
    :param wmbits: n_frames * N矩阵
    :param wm: N维数组
    """
    wm = np.zeros_like(wmbits[0, :])
    for i in range(wm.shape[0]):
        arr = wmbits[:, i]
        tu = sorted([(np.sum(arr == j), j) for j in set(arr)])
        wm[i] = tu[-1][1]
    return wm