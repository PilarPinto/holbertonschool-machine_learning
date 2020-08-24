#!/usr/bin/env python3
'''File of precision'''
import numpy as np


def precision(confusion):
    '''Precision definition'''
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0)
    prec = tp / fp
    return(prec)
