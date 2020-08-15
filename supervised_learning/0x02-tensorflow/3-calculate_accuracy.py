#!/usr/bin/env python3
'''calculates the accuracy of a prediction'''

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    '''calculates the accuracy of a prediction'''
    equival = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    minimun = tf.reduce_mean(tf.cast(equival, tf.float32))
    return minimun
