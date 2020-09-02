#!/usr/bin/env python3
'''File for NN prediction'''
import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''Makes a prediction using a neural network'''
    pr = network.predict(data, verbose=verbose)
    return pr
