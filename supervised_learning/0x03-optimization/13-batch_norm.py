#!/usr/bin/env python3
''' normalizes an unactivated output of a neural network file'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''Normalizes an unactivated output of a neural network'''
    m = Z.mean(0)
    v = Z.var(0)

    Znew = (Z - m) / (v + epsilon)**(1/2)
    normZ = gamma * Znew + beta

    return normZ
