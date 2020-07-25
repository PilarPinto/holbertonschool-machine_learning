#!/usr/bin/env python3
'''Integral of a polynomial'''


def poly_integral(poly, C=0):
    '''Polynomial integrate'''

    if type(poly) is not list or len(poly) is 0 or type(C) is not int:
        return None

    if poly == [0]:
        return [C]

    intg_list = []
    intg_list.append(C)

    if (isinstance(poly, list) and len(poly) > 0):
        for index, item in enumerate(poly):
            int_num = 1/(index+1)
            num = int_num*item
            intg_list.append(int(num) if num.is_integer() else num)
        return(intg_list)
