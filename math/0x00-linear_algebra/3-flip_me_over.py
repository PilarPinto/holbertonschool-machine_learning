#!/usr/bin/env python3
'''
Script to transpose a matrix with python
'''


def matrix_transpose(matrix):
    '''Change the j component by i component and so on'''
    transpose = [[matrix[j][i] for j in range(
        len(matrix))] for i in range(len(matrix[0]))]
    return(transpose)
