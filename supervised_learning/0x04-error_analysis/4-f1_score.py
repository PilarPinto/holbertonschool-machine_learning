#!/usr/bin/env python3
'''F1 score file'''
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    '''F1 score definition'''
    pr = precision(confusion)
    sen = sensitivity(confusion)

    f_scr = 2 * ((pr * sen)/(pr + sen))
    return f_scr
