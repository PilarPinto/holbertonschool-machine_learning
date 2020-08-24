#!/usr/bin/env python3
'''Momentum file'''
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''Momentum definition'''
    n_moment = (beta1 * v) + ((1 - beta1) * grad)
    up_var = var - (alpha * n_moment)
    return up_var, n_moment
