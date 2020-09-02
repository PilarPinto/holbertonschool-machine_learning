#!/usr/bin/env python3
'''that returns one hot matrix with keras'''
import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''Definition that returns one hot matrix with keras'''
    one_hot_matrix = K.utils.to_categorical(labels, classes)
    return one_hot_matrix
