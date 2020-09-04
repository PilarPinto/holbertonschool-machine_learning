#!/usr/bin/env python3
'''Basic convolution file'''
import matplotlib.pyplot as plt
import numpy as np


def convolve_grayscale_valid(images, kernel):
    '''Applied the filter matrix to a img'''
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    h_conv = h - kh + 1
    w_conv = w - kw + 1
    convol_img = np.zeros((m, h_conv, w_conv))

    image = np.arange(0, m)
    for i in range(h_conv):
        for j in range(w_conv):
            img_wise = images[image, i:kh+i, j:kw+j] * kernel
            out_coord = np.sum(img_wise, axis=(1, 2))
            convol_img[image, i, j] = out_coord

    return convol_img
