"""
Created on 11th Sept. 2019

@author: Wushiqiang
"""

import numpy as np
from scipy import linalg


def extract(Y, P):
    """
    将水印信息提取出来
    :param Y: 包含水印信息的频域信息
    :param p0: 伪随机码组成的种子信息
    :return wmbits: 嵌入的水印信息
    """
    N = Y.shape[0]
    wmbits = np.zeros(N, dtype=np.int_)
    
    for j in range(N):
        tmp = np.abs(np.dot(Y[j, :], P.T))
        if tmp.all() == 0:
            return None
        else:
            wmbits[j] = tmp.argmax()
    return wmbits


def pn_reconstruct(n, p0):
    """
    根据保存的种子信息，重建伪随机码字矩阵
    :param p0: 种子序列
    :param n: 需要的随机码字个数
    :return P: 伪随机码字矩阵
    """
    L_n = p0.shape[0]
    mat = np.zeros([L_n, L_n])
    for i in range(L_n):
        mat[i, :] = np.roll(p0, i)
    
    # P = linalg.orth(mat)
    P = np.zeros_like(mat)
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
    return P[:n, :]


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

