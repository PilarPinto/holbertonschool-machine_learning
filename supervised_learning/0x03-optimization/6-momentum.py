#!/usr/bin/env python3
'''Momentum upgraded file'''
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    '''Upgrade the momentum definition'''
    optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(
        loss, global_step=batch)
    return optimizer
