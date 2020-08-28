#!/usr/bin/env python3
'''File that creates a tensorflow layer'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''creates a tensorflow layer'''

    act = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2_reg = tf.contrib.layers.l2_regularizer(lambtha)
    tf_layer = tf.layers.Dense(
        units=n, activation=activation,
        kernel_initializer=act, kernel_regularizer=l2_reg)
    return tf_layer(prev)
