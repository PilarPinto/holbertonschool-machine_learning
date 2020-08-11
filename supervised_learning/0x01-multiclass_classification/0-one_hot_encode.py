#!/usr/bin/env python3
'''one-hot encode file'''
import numpy as np


def one_hot_encode(Y, classes):
    '''one hot encode definition'''
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    for ind in Y:
        if ind >= classes:
            return None

    h_encode = np.zeros((classes, Y.shape[0]))
    h_encode[Y, np.arange(Y.size)] = 1
    return h_encode
