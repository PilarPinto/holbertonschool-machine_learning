#!/usr/bin/env python3
'''
Take the shape of the matrix
'''


def matrix_shape(matrix):
    shape_m = []
    shape_m.append(len(matrix))
    if type(matrix[0]) == list:
        while type(matrix[0]) != int:
            shape_m.append(len(matrix[0]))
            matrix = matrix[0]
    return shape_m
