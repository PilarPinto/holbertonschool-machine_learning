#!/usr/bin/env python3
'''File of poisson probability distribution'''


class Poisson:
    '''
    Class poisson find mean
    '''
    def __init__(self, data=None, lambtha=1.):

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
                lambtha = sum(data) / len(data)
                self.lambtha = float(lambtha)

    def pmf(self, k):
        '''Definition of the PMF formula with lambtha'''
        e = 2.71828
        facto_k = 1
        k = int(k)
        if k <= 0:
            return 0
        for ind in range(1, k+1):
            facto_k = facto_k * ind
        pmf = ((e**-self.lambtha)*(self.lambtha**k))/facto_k
        return pmf
