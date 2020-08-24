#!/usr/bin/env python3
'''File of the normalization constants'''
import numpy as np


def normalization_constants(X):
    '''Normalization definition'''
    avg = np.sum(X, axis=0) / X.shape[0]
    var = np.sum((X - avg)**2, axis=0) / X.shape[0]
    std = np.sqrt(var)
    return(avg, std)
