#!/usr/bin/env python3
'''File to convolve with colors'''
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    '''performs a convolution on images with channels:'''
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
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
    convol_img = np.zeros((m, h_conv, w_conv))

    image = np.arange(0, m)
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)], 'constant',
                    constant_values=0)

    for i in range(h_conv):
        for j in range(w_conv):
            img_wise = images[image, i*sh:kh+(i*sh), j*sw:kw+(j*sw)] * kernel
            out_coord = np.sum(img_wise, axis=(1, 2, 3))
            convol_img[image, i, j] = out_coord

    return convol_img
