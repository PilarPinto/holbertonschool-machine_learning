#!/usr/bin/env python3
'''builds a dense block File'''
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    '''builds a dense block '''
    kernel_init = K.initializers.he_normal(seed=None)

    for layer in range(layers):
        batch1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation('relu')(batch1)
        conv1x1 = K.layers.Conv2D(filters=4*growth_rate, kernel_size=1,
                                  padding='same',
                                  kernel_initializer=kernel_init)(act1)

        batch2 = K.layers.BatchNormalization()(conv1x1)
        act2 = K.layers.Activation('relu')(batch2)
        conv33 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                 padding='same',
                                 kernel_initializer=kernel_init)(act2)

        X = K.layers.concatenate([X, conv33])
        nb_filters += growth_rate

    return X, nb_filters
