#!/usr/bin/env python3
'''File of Regularization Cost with tf'''
import tensorflow as tf


def l2_reg_cost(cost):
    '''Regularization cost with tf'''

    reg_cost = cost + tf.losses.get_regularization_losses(scope=None)
    return reg_cost
