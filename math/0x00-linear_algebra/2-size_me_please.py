#!/usr/bin/env python3
'''
Take the shape of the matrix
'''
def matrix_shape(matrix):
    shape_m = []
    shape_m.append(len(matrix))
    if type(matrix[0]) == list:
        shape_m.append(len(matrix[0]))
        if type(matrix[0][0]) == list:
            shape_m.append(len(matrix[0][0]))

    return shape_m
