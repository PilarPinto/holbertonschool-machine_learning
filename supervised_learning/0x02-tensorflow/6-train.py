#!/usr/bin/env python3
'''Actions neural network classifier file'''

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    '''
    define that builds, trains, and saves a neural network
    '''
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    accur = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accur', accur)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ind in range(iterations + 1):
            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_accur = sess.run(accur, feed_dict={x: X_train, y: Y_train})
            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_accur = sess.run(accur, feed_dict={x: X_valid, y: Y_valid})

            if (ind % 100 == 0) or (ind == iterations) or (ind == 0):
                print('After {} iterations:'.format(ind))
                print('\tTraining Cost: {}'.format(t_cost))
                print('\tTraining Accuracy: {}'.format(t_accur))
                print('\tValidation Cost: {}'.format(v_cost))
                print('\tValidation Accuracy: {}'.format(v_accur))
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        save_path = saver.save(sess, save_path)
    return path
