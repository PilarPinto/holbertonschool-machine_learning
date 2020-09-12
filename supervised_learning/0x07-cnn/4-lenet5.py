#!/usr/bin/env ptyhon3
''' builds a modified version of the LeNet-5 architecture File'''
import tensorflow as tf


def lenet5(x, y):
    '''
    builds a modified version of the
    LeNet-5 architecture using tensorflow
    '''
    ker_init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    conv_lay1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                                 padding='same', activation=activation,
                                 kernel_initializer=ker_init)(x)

    po1 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                 strides=2)(conv_lay1)

    conv_lay2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                                 padding='valid',
                                 activation=activation,
                                 kernel_initializer=ker_init)(po1)

    po2 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                 strides=2)(conv_lay2)

    flatt = tf.layers.Flatten()(po2)

    conv_lay3 = tf.layers.Dense(units=120, activation=activation,
                                kernel_initializer=ker_init)(flatt)

    conv_lay4 = tf.layers.Dense(units=84, activation=activation,
                                kernel_initializer=ker_init)(conv_lay3)

    out_lay = tf.layers.Dense(units=10,
                              kernel_initializer=ker_init)(conv_lay4)

    out = tf.nn.softmax(out_lay)

    loss = tf.losses.softmax_cross_entropy(y, out_lay)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    predict = tf.equal(tf.argmax(y, axis=1),
                       tf.argmax(out_lay, axis=1))
    accur = tf.reduce_mean(tf.cast(predict, tf.float32))

    return out, optimizer, loss, accur
