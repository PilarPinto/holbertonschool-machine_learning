#!/usr/bin/env python3
'''Build a NN with Keras file'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''build a NN with keras definition'''
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)
    model.add(K.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer=reg, name='dense'))
    for l in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[l], activation=activations[l],
                                 kernel_regularizer=reg,
                                 name='dense_' + str(l)))
    return model
