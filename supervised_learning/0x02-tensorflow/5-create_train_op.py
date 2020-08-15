#!/usr/bin/env python3
'''creates the training operation file'''

import tensorflow as tf


def create_train_op(loss, alpha):
    '''training operation def'''
    gradient_train = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return gradient_train
