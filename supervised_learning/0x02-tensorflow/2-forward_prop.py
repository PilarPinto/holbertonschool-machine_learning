#!/usr/bin/env python3
'''Foward propagation'''

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    '''Foward propagation'''
    new_layer = create_layer(x, layer_sizes[0], activations[0])
    for ind in range(1, len(layer_sizes)):
        new_layer = create_layer(new_layer, layer_sizes[ind], activations[ind])
    return new_layer
