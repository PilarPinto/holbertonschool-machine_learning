#!/usr/bin/env python3
'''File to build a projection block'''
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    '''builds a projection block '''
    F11, F3, F12 = filters

    activation = 'relu'
    kernel_init = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                            padding='same',
                            kernel_initializer=kernel_init)(A_prev)

    batch1 = K.layers.BatchNormalization(axis=3)(conv1)

    act1 = K.layers.Activation('relu')(batch1)

    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer=kernel_init)(act1)

    batch2 = K.layers.BatchNormalization(axis=3)(conv2)

    act2 = K.layers.Activation('relu')(batch2)

    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=kernel_init)(act2)

    batch3 = K.layers.BatchNormalization(axis=3)(conv3)

    conv4 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(s, s),
                            padding='same',
                            kernel_initializer=kernel_init)(A_prev)

    batch4 = K.layers.BatchNormalization(axis=3)(conv4)

    add_lay = K.layers.Add()([batch3, batch4])

    activ_out = K.layers.Activation('relu')(add_lay)

    return activ_out
