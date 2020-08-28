#!/usr/bin/env python3
'''File creates a NN using dropout'''
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''create a NN layer using dropout:'''
    act = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop_reg = tf.layers.Dropout(keep_prob)
    tf_layer = tf.layers.Dense(
        units=n, activation=activation,
        kernel_initializer=act, kernel_regularizer=drop_reg)
    return tf_layer(prev)
