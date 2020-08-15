#!/usr/bin/env python3
'''File calculates the softmax cross-entropy loss'''

import tensorflow as tf


def calculate_loss(y, y_pred):
    '''def calculate_loss(y, y_pred):'''
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
