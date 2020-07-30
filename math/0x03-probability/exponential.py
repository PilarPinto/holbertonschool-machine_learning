#!/usr/bin/env python3
'''File of Exponential probability distribution'''


class Exponential:
    '''
    Class Exponential
    '''
    def __init__(self, data=None, lambtha=1.):
        '''Class constructor'''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) <= 2:
                raise ValueError('data must contain multiple values')
            else:
                lambtha = 1 / (sum(data) / len(data))
                self.lambtha = float(lambtha)
