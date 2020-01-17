"""
Created On 10th Sept. 2019

@author: Wushiqiang
"""

import numpy as np


def seed_generate(L_n):
    """
    """
    tmp = np.random.randn(L_n)
    p0 = np.sign(tmp)
    return p0

def pn_code_generate(n, p0):
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
    
    # P = linalg.orth(mat)
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


def alpha(upper, lower, c, N):
    """
    生成按照频谱高低递减的码字权重
    :param upper: 上界
    :param lower: 下界
    :param c: 常数参数
    :param N: 参数个数
    :return a: 参数序列
    """
    a = np.zeros(N)
    for i in range(N):
        a[i] = (upper - lower) * np.exp(-c * i) + lower
    return a


def watermark_embed(X_m, P, wmbits, N=32, weight=1):
    """
    将水印信息嵌入到频谱中去
    :param X_m: 中频信号
    :param P: 伪随机码字矩阵
    :param N: 码字个数
    :param wmbits: 待嵌入码字的信息
    :return Y: 嵌入水印之后的中频频谱信号
    """
    
    upper = 0.15
    lower = 0.075
    c = 0.002
    a = alpha(upper, lower, c, N)
    Y = np.zeros_like(X_m)
    for i in range(N):
        if type(wmbits) is int:
            p_t = P[wmbits, :]
        else:
            p_t = P[wmbits[i], :]
        xxx = np.dot(X_m[i, :], p_t)
        Y[i, :] = X_m[i, :] + weight * a[i] * np.sign(xxx) * p_t
        # a = alpha_update(X_m[i, :])
        # Y[i, :] = X_m[i, :] + a * np.sign(xxx) * p_t
    return Y


def snr(original, embedded):
    """
    计算嵌入的水印的信噪比
    :param original: 原始信号
    :param embedded: 水印信号
    :return snr: 信噪比
    """
    wm = embedded - original
    xpower = np.sum(original ** 2)
    npower = np.sum(wm ** 2)
    snr = xpower / npower
    return 10*np.log10(snr)
    

