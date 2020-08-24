#!/usr/bin/env python3
'''Create confusion'''


def create_confusion_matrix(labels, logits):
    '''Confusion matrix'''
    rta = np.matmul(labels.T, logits)
    return(rta)
