#!/usr/bin/env python3
'''File to pooling'''
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    ''' performs pooling on images'''

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh, sw = stride

    h_conv = int(((h - kh) / sh) + 1)
    w_conv = int(((w - kw) / sw) + 1)

    convol_img = np.zeros((m, h_conv, w_conv, c))
    image = np.arange(0, m)

    for i in range(h_conv):
        for j in range(w_conv):
            img_wise = images[image, i*sh:kh+(i*sh), j*sw:kw+(j*sw)]

            if mode == 'max':
                out_coord = np.max(img_wise, axis=(1, 2))

            if mode == 'avg':
                out_coord = np.mean(img_wise, axis=(1, 2))

            convol_img[image, i, j, ] = out_coord

    return convol_img
