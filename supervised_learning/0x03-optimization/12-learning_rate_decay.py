#!/usr/bin/env python3
'''creates a learning rate decay operation file'''
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Definition rate decay with tf'''
    r_d = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                      decay_rate, True)
    return r_d
