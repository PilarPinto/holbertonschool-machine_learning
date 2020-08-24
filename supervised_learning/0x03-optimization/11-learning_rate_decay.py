#!/usr/bin/env python3
''' updates the learning rate file'''


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Definition that update the learning rate'''
    alpha /= (1 + decay_rate * (global_step // decay_step))
    return alpha
