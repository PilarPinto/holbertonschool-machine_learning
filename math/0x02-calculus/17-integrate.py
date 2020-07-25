#!/usr/bin/env python3
'''Integral of a polynomial'''


def poly_integral(poly, C=0):
    '''Polynomial integrate'''

    if type(poly) is not list or len(poly) == 0 or type(C) is not int:
        return None

    intg_list = [C]

    for index in range(len(poly)):
        if type(poly[index]) is not int and type(poly[index]) is not float:
            return None
        if sum(poly) == 0:
            continue

        num = (1/(index+1))*(poly[index])
        intg_list.append(int(num) if num % 1 == 0 else num)
    return(intg_list)
