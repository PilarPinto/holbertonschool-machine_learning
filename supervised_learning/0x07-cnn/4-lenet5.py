#!/usr/bin/env python3
''' builds a modified version of the LeNet-5 architecture File'''

import tensorflow as tf


def lenet5(x, y):
    '''LeNet-5 architecture using tensorflow'''

    k_nit = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    layer_1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                               activation=activation,
                               kernel_initializer=k_init)(x)

    pool_1 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                    strides=2)(layer_1)

    layer_2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                               padding='valid',
                               activation=activation,
                               kernel_initializer=k_init)(pool_1)

    pool_2 = tf.layers.MaxPooling2D(pool_size=2,
                                    strides=2)(layer_2)

    flatten = tf.layers.Flatten()(pool_2)

    layer_3 = tf.layers.Dense(units=120, activation=activation,
                              kernel_initializer=k_init)(flatten)

    layer_4 = tf.layers.Dense(units=84, activation=activation,
                              kernel_initializer=k_init)(layer_3)

    output_layer = tf.layers.Dense(units=10,
                                   kernel_initializer=k_init)(layer_4)

    y_out = tf.nn.softmax(output_layer)

    equality = tf.equal(tf.argmax(y,1),
                        tf.argmax(output_layer, axis=1))
    acc = tf.reduce_mean(tf.cast(equality, tf.float32))

    loss = tf.losses.softmax_cross_entropy(y, output_layer)

    train = tf.train.AdamOptimizer().minimize(loss)

    return(y_out, train, loss, acc)
