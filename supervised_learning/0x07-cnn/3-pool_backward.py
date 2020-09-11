#!/usr/bin/env python3
'''performs back propagation over a pooling'''
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    ''' performs back propagation over a pooling layer '''
    m, h_new,  w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    for ind_m in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for ci in range(c_new):
                    prev = A_prev[ind_m, i*sh:kh+(i*sh), j*sw:kw+(j*sw), ci]
                    filter = (prev == np.max(prev))
                    if mode == 'max':
                        dA_prev[ind_m, i*sh:kh+(i*sh),
                                j*sw:kw+(j*sw), ci] += dA[ind_m,
                                                          i, j, ci] * filter
                    if mode == 'avg':
                        dA_prev[ind_m, i*sh:kh+(i*sh),
                                j*sw:kw+(j*sw), ci] += (dA[ind_m,
                                                           j, w, ci])/kh/kw
    return dA_prev
