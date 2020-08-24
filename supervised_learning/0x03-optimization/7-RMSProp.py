#!/usr/bin/env python3
'''RMSProp file'''


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''Update RSM'''
    n_moment = (beta2 * s) + ((1 - beta2) * grad**2)
    s_m = grad / ((n_moment**0.5) + epsilon)
    up_var = var - (alpha * s_m)
    return(up_var, n_moment)
