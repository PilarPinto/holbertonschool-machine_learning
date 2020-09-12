#!/usr/bin/env python3
'''performs back propagation over a pooling'''
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    ''' performs back propagation over a pooling layer '''
    m, h_new,  w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_convol = int(float(h_prev - kh) / float(sh)) + 1
    w_convol = int(float(w_prev - kw) / float(sw)) + 1

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for y in range(h_convol):
            for x in range(w_convol):
                for ch in range(c):
                    A = dA[i, y, x, ch]
                    image = A_prev[i, y * sh:y * sh + kh,
                                   x * sw:x * sw + kw, ch]
                    if mode == 'max':
                        res = (image == np.max(image))
                        dA_prev[i, y * sh:y * sh + kh,
                                x * sw:x * sw + kw, ch] += A * res
                    elif mode == 'avg':
                        res = A / kh / kw
                        dA_prev[i, y * sh:y * sh + kh,
                                x * sw:x * sw + kw, ch] += np.ones((
                                    kh, kw)) * res
    return(dA_prev)
