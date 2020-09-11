#!/usr/bin/env python3
'''builds a modified version of the LeNet-5 arch keras '''

import numpy as np
import tensorflow as tf
import tensorflow.keras as K


def lenet5(X):
    '''builds a modified version of the LeNet-5 architecture using keras'''
    activation = 'relu'
    ker_init = K.initializers.he_normal(seed=None)

    conv_lay1 = K.layers.Conv2D(filters=6, kernel_size=5,
                                padding='same', activation=activation,
                                kernel_initializer=ker_init)(X)

    po1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_lay1)

    conv_lay2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                                activation=activation,
                                kernel_initializer=ker_init)(po1)

    po2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_lay2)

    flatt = K.layers.Flatten()(po2)

    conv_lay3 = K.layers.Dense(120, activation=activation,
                               kernel_initializer=ker_init)(flatt)

    conv_lay4 = K.layers.Dense(84, activation=activation,
                               kernel_initializer=ker_init)(conv_lay3)

    output_layer = K.layers.Dense(10, activation='softmax',
                                  kernel_initializer=ker_init)(conv_lay4)

    model = K.models.Model(X, output_layer)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
