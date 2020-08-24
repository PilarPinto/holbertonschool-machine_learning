#!/usr/bin/env python3
'''RMSProp optimizer file'''
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''RMSPRop optimize'''
    optimizer = tf.train.RMSPropOptimizer(alpha,
                                          decay=beta2,
                                          epsilon=epsilon).minimize(loss)
    return optimizer
