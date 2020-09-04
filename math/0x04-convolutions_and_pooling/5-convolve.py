#!/usr/bin/env python3
'''Convolve with multiple kernels diles'''
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    '''performs a convolution on images using multiple kernels'''

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    if padding == 'valid':
        ph, pw = 0, 0
    if type(padding) is tuple:
        ph, pw = padding

    h_conv = int(((h - kh + (2 * ph)) / sh) + 1)
    w_conv = int(((w - kw + (2 * pw)) / sw) + 1)
    convol_img = np.zeros((m, h_conv, w_conv, nc))

    image = np.arange(0, m)
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)], 'constant',
                    constant_values=0)

    for i in range(h_conv):
        for j in range(w_conv):
            for channel in range(nc):
                img_wise = images[image, i*sh:kh+(i*sh),
                                  j*sw:kw+(j*sw)] * kernels[:, :, :, channel]
                out_coord = np.sum(img_wise, axis=(1, 2, 3))
                convol_img[image, i, j, channel] = out_coord

    return convol_img
