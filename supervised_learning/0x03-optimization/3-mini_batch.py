#!/usr/bin/env python3
'''Mini batch file'''

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def batches(t, b_size):
    '''Divide total data in segments'''
    batch_lst = []
    ind = 0
    m = t.shape[0]
    batch_r = int(m / b_size) + (m % b_size > 0)

    for i in range(batch_r):
        if i != batch_r - 1:
            batch_lst.append(t[ind:(ind + b_size)])
        else:
            batch_lst.append(t[ind:])
        ind += b_size
    return batch_lst


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    '''Divide GD in batches'''
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs + 1):
            X, Y = shuffle_data(X_train, Y_train)

            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_accuracy = sess.run(accuracy, feed_dict={x: X_train,
                                                       y: Y_train})

            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_accuracy = sess.run(accuracy, feed_dict={x: X_valid,
                                                       y: Y_valid})

            print("After {} epochs".format(i))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_accuracy))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_accuracy))

            if i < epochs:
                X_b = batches(X, batch_size)
                Y_b = batches(Y, batch_size)

                for b in range(1, len(X_b) + 1):
                    sess.run(train_op, feed_dict={x: X_b[b - 1],
                                                  y: Y_b[b - 1]})

                    t_cost, t_accuracy = sess.run((loss, accuracy),
                                                  feed_dict={x: X_b[b - 1],
                                                             y: Y_b[b - 1]})

                    if not b % 100:
                        print('\tStep {}:'.format(b))
                        print('\t\tCost: {}'.format(t_cost))
                        print('\t\tAccuracy: {}'.format(t_accuracy))

        save_path = saver.save(sess, save_path)
    return save_path
