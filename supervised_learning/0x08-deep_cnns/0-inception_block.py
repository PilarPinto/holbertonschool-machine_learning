#!/usr/bin/env python3
'''builds an inception block as GoogleNet'''

import tensorflow.keras as K


def inception_block(A_prev, filters):
    '''builds an inception block '''
    F1, F3R, F3, F5R, F5, FPP = filters
    activation = 'relu'

    conv11 = K.layers.Conv2D(filters=F1, kernel_size=1,
                             activation=activation)(A_prev)

    conv13 = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                             activation=activation)(A_prev)

    conv33 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                             activation=activation)(conv13)

    conv15 = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                             activation=activation)(A_prev)

    conv55 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                             activation=activation)(conv15)

    lay_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=1,
                                     padding='same')(A_prev)

    conv11_poolR = K.layers.Conv2D(filters=FPP, kernel_size=1,
                                   padding='same',
                                   activation=activation)(lay_pool)

    lst = [conv11, conv33, conv55, conv11_poolR]
    concat_output = K.layers.concatenate(lst)
    return concat_output
