#!/usr/bin/env python3
'''Adam upgraded file'''
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''Adam upgraded def'''
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1, beta2=beta2,
                                       epsilon=epsilon).minimize(loss)
    return optimizer
