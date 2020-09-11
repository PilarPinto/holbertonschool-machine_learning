#!/usr/bin/env python3
'''File of foward convolution'''
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''forward propagation over a convolutional layer of a neural network'''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev * sh) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev * sw) - sw + kw - w_prev) / 2))

    if padding == 'valid':
        ph, pw = 0, 0

    h_conv = int(((h_prev - kh + (2 * ph)) / sh) + 1)
    w_conv = int(((w_prev - kw + (2 * pw)) / sw) + 1)
    convol_img = np.zeros((m, h_conv, w_conv, c_new))

    image = np.arange(0, m)
    image_pad = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                       mode='constant')

    for i in range(h_conv):
        for j in range(w_conv):
            for cn in range(c_new):
                img_wise = image_pad[image, i*sh:kh+(i*sh),
                                     j*sw:kw+(j*sw)] * W[:, :, :, cn]
                out_coord = np.sum(img_wise, axis=(1, 2, 3))
                convol_img[image, i, j, cn] = activation((out_coord
                                                          + b[..., cn]))

    return convol_img
