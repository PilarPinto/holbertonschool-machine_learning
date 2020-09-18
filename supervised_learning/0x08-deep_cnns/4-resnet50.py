#!/usr/bin/env python3
'''builds the ResNet-50 architecture file'''
import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    ''' builds the ResNet-50 architecture'''
    activation = 'relu'
    kernel_init = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same', kernel_initializer=kernel_init)(X)

    batch1 = K.layers.BatchNormalization(axis=3)(conv1)

    act1 = K.layers.Activation('relu')(batch1)

    l1_pool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                 padding='same')(act1)

    pro_block1 = projection_block(l1_pool, [64, 64, 256], 1)

    id_block1 = identity_block(pro_block1, [64, 64, 256])
    id_block2 = identity_block(id_block1, [64, 64, 256])

    pro_block2 = projection_block(id_block2, [128, 128, 512])

    id_block3 = identity_block(pro_block2, [128, 128, 512])
    id_block4 = identity_block(id_block3, [128, 128, 512])
    id_block5 = identity_block(id_block4, [128, 128, 512])

    pro_block3 = projection_block(id_block5, [256, 256, 1024])

    id_block6 = identity_block(pro_block3, [256, 256, 1024])
    id_block7 = identity_block(id_block6, [256, 256, 1024])
    id_block8 = identity_block(id_block7, [256, 256, 1024])
    id_block9 = identity_block(id_block8, [256, 256, 1024])
    id_block10 = identity_block(id_block9, [256, 256, 1024])

    pro_block4 = projection_block(id_block10, [512, 512, 2048])

    id_block11 = identity_block(pro_block4, [512, 512, 2048])
    id_block12 = identity_block(id_block11, [512, 512, 2048])

    avg_p = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(id_block12)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=kernel_init)(avg_p)

    model = K.models.Model(inputs=X, outputs=Y)

    return model
