#!/usr/bin/env python3
'''File for sensivity'''
import numpy as np


def sensitivity(confusion):
    '''Sensivity definition'''
    tp = np.diagonal(confusion)
    pos = np.sum(confusion, axis=1)
    return (tp/pos)
