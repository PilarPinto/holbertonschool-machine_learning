#!/usr/bin/env python3
'''Architecture LeNet-5 file'''
import tensorflow as tf


def lenet5(x, y):
    '''Architecture t-net'''
    # kernel initialized with he_normal method
    kernel = tf.contrib.layers.variance_scaling_initializer()

    layer1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding="same",
                              kernel_initializer=kernel,
                              activation=tf.nn.relu)(x)

    max1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(layer1)

    layer2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding="valid",
                              kernel_initializer=kernel,
                              activation=tf.nn.relu)(max1)

    max2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(layer2)

    flatt = tf.layers.Flatten()(max2)

    layer3 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                             kernel_initializer=kernel)(flatt)
    layer4 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                             kernel_initializer=kernel)(layer3)
    output_layer = tf.layers.Dense(units=10,
                                   kernel_initializer=kernel)(layer4)
    output_act = tf.nn.softmax(output_layer)

    prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    loss = tf.losses.softmax_cross_entropy(y, output_layer)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return(output_act, optimizer, loss, accuracy)
