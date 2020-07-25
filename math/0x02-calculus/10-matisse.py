#!/usr/bin/env python3
'''Derivate of a polynomial'''


def poly_derivative(poly):
    '''Polynomial derivate'''
    der_list = []
    if (len(poly) == 1):
        return('[0]')
    if (isinstance(poly, list) and len(poly) > 0):
        for index, item in enumerate(poly):
            num = index*item
            if index != 0:
                der_list.append(num)
        return(der_list)
    else:
        return None
