#!/usr/bin/env python3
'''Mini batch file'''

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


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

        mini_b = X_train.shape[0] / batch_size
        if type(mini_b) is not int:
            mini_b = int(mini_b + 1)

        for i in range(epochs + 1):
            t_cst, t_acc = sess.run([loss, accuracy], feed_dict={x: X_train,
                                                                 y: Y_train})

            v_cst, v_acc = sess.run([loss, accuracy], feed_dict={x: X_valid,
                                                                 y: Y_valid})

            print("After {} epochs".format(i))
            print("\tTraining Cost: {}".format(t_cst))
            print("\tTraining Accuracy: {}".format(t_acc))
            print("\tValidation Cost: {}".format(v_cst))
            print("\tValidation Accuracy: {}".format(v_acc))

            if i < epochs:
                xs, ys = shuffle_data(X_train, Y_train)

                for b in range(1, mini_b + 1):
                    ft = (b - 1) * batch_size
                    lt = b * batch_size
                    if lt > X_train.shape[0]:
                        lt = X_train.shape[0]
                    batch = {x: xs[ft:lt], y: ys[ft:lt]}
                    sess.run(train_op, feed_dict=batch)

                    if b % 100 is 0:
                        cst, accu = sess.run((loss, accuracy), feed_dict=batch)
                        print('\tStep {}:'.format(b))
                        print('\t\tCost: {}'.format(cst))
                        print('\t\tAccuracy: {}'.format(accu))

        save_path = saver.save(sess, save_path)

    return(save_path)
