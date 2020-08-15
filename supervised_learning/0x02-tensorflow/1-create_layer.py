#!/usr/bin/env python3
'''Create a layer'''

import tensorflow as tf


def create_layer(prev, n, activation):
    '''Create a layer'''
    act = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=act, name='layer')
    return layer(prev)
