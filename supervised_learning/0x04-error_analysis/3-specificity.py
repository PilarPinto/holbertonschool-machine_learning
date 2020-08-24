#!/usr/bin/env python3
'''specificity file'''
import numpy as np


def specificity(confusion):
    '''specificity definition'''
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=1) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = np.sum(confusion) - (fp + fn + tp)
    tnr = tn/(tn+fp)
    return tnr
