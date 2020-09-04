#!/usr/bin/env python3
''' performs a same convolution on grayscale images'''
import matplotlib.pyplot as plt
import numpy as np


def convolve_grayscale_same(images, kernel):
    '''performs a same convolution on grayscale images'''
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    if not kh % 2:
        ph = int(kh / 2)
        h_conv = h - kh + (2 * ph)
    else:
        ph = int((kh - 1) / 2)
        h_conv = h - kh + 1 + (2 * ph)

    if not kw % 2:
        pw = int(kw / 2)
        w_conv = w - kw + (2 * pw)
    else:
        pw = int((kw - 1) / 2)
        w_conv = w - kw + 1 + (2 * pw)

    convol_img = np.zeros((m, h_conv, w_conv))
    image = np.arange(0, m)
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant',
                    constant_values=0)

    for i in range(h_conv):
        for j in range(w_conv):
            img_wise = images[image, i:kh+i, j:kw+j] * kernel
            out_coord = np.sum(img_wise, axis=(1, 2))
            convol_img[image, i, j] = out_coord

    return convol_img
