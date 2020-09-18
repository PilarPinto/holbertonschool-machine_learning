#!/usr/bin/env python3
'''builds the inception network file'''

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    '''builds the inception network '''
    activation = 'relu'

    X = K.Input(shape=(224, 224, 3))

    convl1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                             padding='same', activation=activation)(X)

    l1_pool = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                    padding='same')(convl1)

    convl2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                             activation=activation)(l1_pool)

    convl3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same',
                             activation=activation)(convl2)

    l2_pool = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                    padding='same')(convl3)

    in_block1 = inception_block(l2_pool, [64, 96, 128, 16, 32, 32])

    in_block2 = inception_block(in_block1, [128, 128, 192, 32, 96, 64])

    l3_pool = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                    padding='same')(in_block2)

    in_block3 = inception_block(l3_pool, [192, 96, 208, 16, 48, 64])

    in_block4 = inception_block(in_block3, [160, 112, 224, 24, 64, 64])

    in_block5 = inception_block(in_block4, [128, 128, 256, 24, 64, 64])

    in_block6 = inception_block(in_block5, [112, 144, 288, 32, 64, 64])

    in_block7 = inception_block(in_block6, [256, 160, 320, 32, 128, 128])

    l4_pool = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                    padding='same')(in_block7)

    in_block8 = inception_block(l4_pool, [256, 160, 320, 32, 128, 128])

    in_block9 = inception_block(in_block8, [384, 192, 384, 48, 128, 128])

    avg_p = K.layers.AveragePooling2D(pool_size=[7, 7], strides=(7, 7),
                                      padding='same')(in_block9)

    dropout = K.layers.Dropout(0.4)(avg_p)

    Y = K.layers.Dense(1000, activation='softmax')(dropout)

    model = K.models.Model(inputs=X, outputs=Y)
    return model
