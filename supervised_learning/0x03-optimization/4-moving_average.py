#!/usr/bin/env python3
'''Moving average file'''


def moving_average(data, beta):
    '''Moving verage def'''
    lst = []
    vt = 0
    for ind in range(len(data)):
        t = ind + 1
        vt = beta * vt + (1 - beta) * data[ind]
        correct = vt / (1 - beta**t)
        lst.append(correct)
    return (lst)
