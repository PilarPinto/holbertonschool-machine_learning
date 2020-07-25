#!/usr/bin/env python3
'''Integral of a polynomial'''


def poly_integral(poly, C=0):
    '''Polynomial integrate'''

    if type(poly) is not list or len(poly) is 0 or type(C) is not int:
        return None

    intg_list = [C]

    for index in range(len(poly)):
        if sum(poly) == 0:
            return intg_list

        if type(poly[index]) is int or type(poly[index]) is float:
            num = (1/(index+1))*(poly[index])
            intg_list.append(int(num) if num % 1 == 0 else num)
        else:
            return None
        return(intg_list)
