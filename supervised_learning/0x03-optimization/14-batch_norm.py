#!/usr/bin/env python3
'''Normalization upgraded file'''
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''Normalization upgraded with tf'''
    ker_i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=ker_i).apply(prev)

    mean, var = tf.nn.moments(layer, axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), True)

    b_norm = tf.nn.batch_normalization(layer, mean, var,
                                       beta, gamma, 1e-8)

    return(activation(b_norm))
