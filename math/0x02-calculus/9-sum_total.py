#!/usr/bin/env python3
'''Sum program'''
import functools


def summation_i_squared(n):
    '''Sum definition'''
    if (isinstance(n, int) and n > 0):
        return(functools.reduce(lambda a, b: a+b, list(
            map(lambda i: i**2, range(n+1)))))
    else:
        return None
