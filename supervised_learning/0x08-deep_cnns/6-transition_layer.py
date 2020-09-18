#!/usr/bin/env python3
'''File to create atransition layer '''
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    '''builds a transition layer '''
    kernel_init = K.initializers.he_normal(seed=None)
    filters = int(nb_filters * compression)

    batch1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation('relu')(batch1)
    trans = K.layers.Conv2D(filters=filters, kernel_size=1, padding='same',
                            kernel_initializer=kernel_init)(act1)

    avg_p = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                      padding='same')(trans)

    return avg_p, filters
