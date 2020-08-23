#!/usr/bin/env python3
'''Shuffle data file'''
import numpy as np


def shuffle_data(X, Y):
    '''Shuffles the data points'''
    sh = np.random.permutation(X.shape[0])
    return (X[sh], Y[sh])
