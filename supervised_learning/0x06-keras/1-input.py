#!/usr/bin/env python3
'''Build NN with Keras File'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Builds a NN with keras input'''
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)
    output = K.layers.Dense(layers[0], input_shape=(nx,),
                            activation=activations[0],
                            kernel_regularizer=reg, name='dense')(inputs)

    for l in range(1, len(layers)):
        drop = K.layers.Dropout(1 - keep_prob)(output)
        output = K.layers.Dense(layers[l], activation=activations[l],
                                kernel_regularizer=reg,
                                name='dense_' + str(l))(drop)

    model = K.models.Model(inputs, output)
    return model
