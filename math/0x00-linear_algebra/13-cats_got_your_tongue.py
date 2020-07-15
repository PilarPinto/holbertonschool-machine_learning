#!/usr/bin/env python3
'''Concatenates two matrices with specific axis'''
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''Using concatenate numpy function'''
    return np.concatenate((mat1, mat2), axis)
