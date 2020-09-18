#!/usr/bin/env python3
'''builds the DenseNet-121 architecture '''


def densenet121(growth_rate=32, compression=1.0):
    '''builds the DenseNet-121 architecture '''
    kernel_init = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))
    batch0 = K.layers.BatchNormalization(axis=3)(X)
    act0 = K.layers.Activation('relu')(batch0)

    conv1 = K.layers.Conv2D(filters=2*growth_rate, kernel_size=(7, 7),
                            strides=(2, 2), padding='same',
                            kernel_initializer=kernel_init)(act0)

    l1_pool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                 padding='same')(conv1)

    den_block1, filters2 = dense_block(l1_pool, 2*growth_rate, growth_rate, 6)
    trans1, filters3 = transition_layer(den_block1, filters2, compression)
    den_block2, filters4 = dense_block(trans1, filters3, growth_rate, 12)
    trans2, filters5 = transition_layer(den_block2, filters4, compression)
    den_block3, filters6 = dense_block(trans2, filters5, growth_rate, 24)
    trans3, filters7 = transition_layer(den_block3, filters6, compression)
    den_block4, filters8 = dense_block(trans3, filters7, growth_rate, 16)

    avg_p = K.layers.AveragePooling2D(pool_size=(7, 7), strides=7,
                                      padding='same')(den_block4)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=kernel_init)(avg_p)

    model = K.models.Model(inputs=X, outputs=Y)

    return model
