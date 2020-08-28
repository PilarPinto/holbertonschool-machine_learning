#!/usr/bin/env python3
''' if you should stop gradient descent early:'''


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''Check condition of stopping'''
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if (count == patience):
        boolean = True
    else:
        boolean = False

    return boolean, count
