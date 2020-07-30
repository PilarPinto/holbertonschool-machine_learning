#!/usr/bin/env python3
'''File of Binomial probability distribution'''


class Binomial:
    '''
    Class Binomial
    '''
    def __init__(self, data=None, n=1, p=0.5):
        '''Contructor of class binomial'''
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p > 1 and p < 0:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) <= 2:
                raise ValueError('data must contain multiple values')
            else:
                mean = sum(data) / len(data)
                sum_vari = 0
                for ind in range(len(data)):
                    sum_vari += (data[ind] - mean)**2
                varianza = sum_vari / len(data)
                p = 1 - (varianza / mean)
                n = mean / p
                n = round(n)
                p = mean / n
                self.n = int(n)
                self.p = float(p)
