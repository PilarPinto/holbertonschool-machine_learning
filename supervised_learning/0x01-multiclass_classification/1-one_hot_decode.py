#!/usr/bin/env python3
'''one-hot decode file'''
import numpy as np


def one_hot_decode(one_hot):
    '''one hot decode definition'''
    if type(one_hot) is not np.ndarray:
        return None
    return np.argmax(one_hot, axis=0)
