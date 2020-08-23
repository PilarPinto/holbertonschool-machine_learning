#!/usr/bin/env python3
'''Normalize a matrix file'''


def normalize(X, m, s):
    '''Standarizes a matrix'''
    Z = (X - m) / s
    return Z
