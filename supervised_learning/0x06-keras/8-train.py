#!/usr/bin/env python3
'''mini-batch gradient descent train file'''
import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    '''Trains a model using mini-batch gradient descent
    early stopping, learning rate decay and validate it'''
    call_lst = []

    def trainer(epoch):
        '''Compute the learning rate'''
        la = alpha / (1 + decay_rate * epoch)
        return la

    if validation_data:
        if early_stopping:
            e_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
            call_lst.append(e_stop)

        if learning_rate_decay:
            learn_rate = K.callbacks.LearningRateScheduler(trainer, verbose=1)
            call_lst.append(learn_rate)

        if save_best:
            save = K.callbacks.ModelCheckpoint(filepath)
            call_lst.append(save)

    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          shuffle=shuffle, verbose=verbose,
                          callbacks=call_lst)

    return history
