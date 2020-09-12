#!/usr/bin/env python3
'''performs back propagation over a convolutional layer'''
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''back propagation over a convolutional layer of a neural network:'''
    m_z, h_new, w_new, c_newz = dZ.shape

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0

    if padding == 'same':
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
        ph = -(-ph // 2)
        pw = -(-pw // 2)

    dW = np.zeros(W.shape)
    dA_prev = np.zeros(A_prev.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i_m in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for cn in range(c_new):
                    W_z = W[..., cn]
                    Z_z = dZ[i_m, i, j, cn]
                    dA_prev[i_m, i * sh:kh + (i * sh), j * sw:kw +
                            (j * sw)] += (W_z * Z_z)
                    dW[..., cn] += (A_prev[i_m, i * sh:kh + (i * sh),
                                           j * sw:kw + (j * sw)] * Z_z)

    _, hd, wd, _ = dA_prev.shape
    dA_prev = dA_prev[:, ph:hd-ph, pw:wd-pw, :]
    return dA_prev, dW, db
