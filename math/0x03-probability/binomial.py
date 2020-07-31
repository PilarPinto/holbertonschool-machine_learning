#!/usr/bin/env python3
'''File of Binomial probability distribution'''


class Binomial:
    '''
    Class Binomial
    '''
    def __init__(self, data=None, n=1, p=0.5):
        '''Contructor of class binomial and definition'''
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p >= 1 or p <= 0:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) < 2:
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

    def pmf(self, k):
        '''Definition of probability density function'''
        k = int(k)
        if k < 0:
            return 0
        n_fac = Binomial.factor(self.n)
        k_fac = Binomial.factor(k)
        nk_fac = Binomial.factor(self.n-k)
        q = 1 - self.p

        bin_coeff = (n_fac)/(k_fac * nk_fac)
        pmf = bin_coeff * (self.p**k) * (q**(self.n-k))
        return pmf

    def factor(number):
        '''Factorial outside of function'''
        facto_k = 1
        for ind in range(1, number + 1):
            facto_k = facto_k * ind
        return facto_k

    def cdf(self, k):
        '''Definition of cumulative density function binomial'''
        k = int(k)
        cdf = 0
        if k < 0:
            return 0
        for ind in range(k + 1):
            cdf += self.pmf(ind)
        return cdf
