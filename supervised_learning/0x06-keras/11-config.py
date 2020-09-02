#!/usr/bin/env python3
'''Optimize file'''
import tensorflow.keras as K


def save_config(network, filename):
    '''saves a model’s configuration in JSON format'''
    json_format = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_format)
    return None


def load_config(filename):
    '''loads a model with a specific configuration'''
    with open(filename, 'r') as f:
        json_format = K.models.model_from_json(f.read())
    return json_format
