#!/usr/bin/env python3
'''Optimize file'''
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''Adam optimization for a keras model'''
    adam = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=adam, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
