#!/usr/bin/env python3
'''mini-batch gradient descent train file'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    '''Trains a model using mini-batch gradient descent
    early stopping and validate it'''
    call_lst = []
    if validation_data:
        e_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=patience)
        call_lst.append(e_stop)
    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          shuffle=shuffle, verbose=verbose,
                          callbacks=call_lst)

    return history
