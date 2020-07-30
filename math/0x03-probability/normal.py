#!/usr/bin/env python3
'''File of normal probability distribution'''


class Normal:
    '''
    Class Normal
    '''
    def __init__(self, data=None, mean=0., stddev=1.):
        '''Constructor of normal class'''
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) <= 2:
                raise ValueError('data must contain multiple values')
            else:
                mean = sum(data) / len(data)
                sigma_sum = 0
                for ind in range(len(data)):
                    sigma_sum += (data[ind] - mean)**2

                sigm = sigma_sum / len(data)
                stddev = sigm**(1/2)
                self.mean = float(mean)
                self.stddev = float(stddev)

    def z_score(self, x):
        '''z-score formula'''
        z_score = (x - self.mean) / (self.stddev)
        return z_score

    def x_value(self, z):
        '''x_value formula'''
        x_value = z * (self.stddev) + self.mean
        return x_value

    def pdf(self, x):
        '''The density function'''
        pi = 3.1415926536
        e = 2.7182818285
        first_div = 1 / (self.stddev * (2 * pi)**(1/2))
        expon = ((x - self.mean)**2/(2*(self.stddev**2)))
        pdf = first_div * (e**(-expon))
        return pdf
