#!/usr/bin/env python3
'''File where builds, trains, and saves a NN'''
import numpy as np
import tensorflow as tf


def create_layer(prev, n, activation):
    '''Create a layer'''
    act = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=act, name='layer')
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    '''Normalization upgraded with tf'''
    if not activation:
        ly = create_layer(prev, n, activation)
        return ly

    ker_i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=ker_i).apply(prev)

    mean, var = tf.nn.moments(layer, axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), True)

    b_norm = tf.nn.batch_normalization(layer, mean, var,
                                       beta, gamma, 1e-8)

    return(activation(b_norm))


def create_placeholders(nx, classes):
    '''Definiton of placeholders'''
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def forward_prop(x, layer_sizes=[], activations=[]):
    '''Foward propagation'''
    new_layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for ind in range(1, len(activations)):
        new_layer = create_batch_norm_layer(new_layer,
                                            layer_sizes[ind], activations[ind])
    return new_layer


def calculate_accuracy(y, y_pred):
    '''calculates the accuracy of a prediction'''
    equival = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    minimun = tf.reduce_mean(tf.cast(equival, tf.float32))
    return minimun


def calculate_loss(y, y_pred):
    '''def calculate_loss(y, y_pred):'''
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''Adam upgraded def'''
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1, beta2=beta2,
                                       epsilon=epsilon).minimize(loss)
    return optimizer


def shuffle_data(X, Y):
    '''Shuffles the data points'''
    sh = np.random.permutation(X.shape[0])
    return (X[sh], Y[sh])


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Definition rate decay with tf'''
    r_d = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                      decay_rate, True)
    return r_d


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    '''definition builds, trains, and saves NN'''

    x, y = create_placeholders(Data_train[0].shape[1], Data_train[1].shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    mini_batch = Data_train[0].shape[0] / batch_size
    if (mini_batch).is_integer() is True:
        mini_batch = int(mini_batch)
    else:
        mini_batch = int(mini_batch + 1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(epochs + 1):
            tc, ta = sess.run([loss, accuracy], feed_dict={x: Data_train[0],
                                                           y: Data_train[1]})
            vc, va = sess.run([loss, accuracy], feed_dict={x: Data_valid[0],
                                                           y: Data_valid[1]})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(tc))
            print("\tTraining Accuracy: {}".format(ta))
            print("\tValidation Cost: {}".format(vc))
            print("\tValidation Accuracy: {}".format(va))

            if i < epochs:
                xs, ys = shuffle_data(Data_train[0], Data_train[1])
                sess.run(global_step.assign(i))
                sess.run(alpha)
                for j in range(1, mini_batch + 1):
                    ft = (j - 1) * batch_size
                    lt = j * batch_size
                    if lt > Data_train[0].shape[0]:
                        lt = Data_train[0].shape[0]
                    batch = {x: xs[ft:lt], y: ys[ft:lt]}
                    sess.run(train_op, feed_dict=batch)
                    if j % 100 is 0:
                        cost = sess.run(loss, feed_dict=batch)
                        accur = sess.run(accuracy, feed_dict=batch)
                        print("\tStep {}:".format(j))
                        print("\t\tCost: {}".format(cost))
                        print("\t\tAccuracy: {}".format(accur))
        save_path = saver.save(sess, save_path)
    return(save_path)
