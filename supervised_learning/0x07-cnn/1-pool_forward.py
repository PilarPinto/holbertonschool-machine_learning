#!/usr/bin/env python3
'''performs forward propagation over a pooling '''
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
    Function that performs forward propagation over a pooling layer of a NN
    '''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_conv = int(((h_prev - kh) / sh) + 1)
    w_conv = int(((w_prev - kw) / sw) + 1)
    convol_img = np.zeros((m, h_conv, w_conv, c_prev))

    image = np.arange(0, m)

    for i in range(h_conv):
        for j in range(w_conv):
            img = A_prev[image, i*sh:kh+(i*sh), j*sw:kw+(j*sw)]

        if mode == 'max':
            out_coord = np.max(img, axis=(1, 2))
        if mode == 'avg':
            out_coord = np.mean(img, axis=(1, 2))

        convol_img[image, i, j] = out_coord

    return convol_img
