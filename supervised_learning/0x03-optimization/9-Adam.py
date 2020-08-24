#!/usr/bin/env python3
'''Adam algorithm file'''

import numpy as np


def update_variables_Adam(alpha, beta1, beta2,
                          epsilon, var, grad, v, s, t):
    '''Adam definition'''

    n_moment = (beta1 * v) + ((1 - beta1) * grad)
    s_moment = (beta2 * s) + ((1 - beta2) * grad**2)

    v_corr = n_moment / (1 - beta1 ** t)
    s_corr = s_moment / (1 - beta2 ** t)

    v_s = v_corr / (np.sqrt(s_corr) + epsilon)
    up_var = var - (alpha * v_s)
    return up_var, n_moment, n_moment
