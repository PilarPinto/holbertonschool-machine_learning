#!/usr/bin/env python3
'''File performs conv with padding'''
import matplotlib.pyplot as plt
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    ''''performs a convolution on grayscale images with custom padding'''
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]

    h_conv = h + (2 * ph) - kh + 1
    w_conv = w + (2 * pw) - kw + 1

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
