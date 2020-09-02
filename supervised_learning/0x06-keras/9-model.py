#!/usr/bin/env python3
'''file wit saving and loading model definitions'''
import tensorflow.keras as K


def save_model(network, filename):
    '''saves an entire model'''
    network.save(filename)
    return None


def load_model(filename):
    '''loads an entire model'''
    load_NN = K.models.load_model(filename)
    return load_NN
