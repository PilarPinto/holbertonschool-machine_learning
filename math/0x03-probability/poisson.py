#!/usr/bin/env python3
'''Poisson file-Probabilty Distribution'''


class Poisson:
    '''
    Class poisson find mean
    '''
    def __init__(self, data=None, lambtha=1.):
        '''Class constructor with lambda and data set'''

        if data is None:
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                lambtha = sum(data) / len(data)
                self.lambtha = float(lambtha)
