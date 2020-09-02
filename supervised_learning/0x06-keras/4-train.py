#!/usr/bin/env python3
'''mini-batch gradient descent train file'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    '''Trains a model using mini-batch gradient descent'''
    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle, verbose=verbose)

    return history
