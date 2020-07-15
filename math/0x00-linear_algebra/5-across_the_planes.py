#!/usr/bin/env python3
'''
Script two add two matix by elements-wise
'''


def matrix_shape(matrix):
    '''Function to obtain a shape of a matrix'''
    shape_m = []
    shape_m.append(len(matrix))
    if type(matrix[0]) == list:
        while type(matrix[0]) != int:
            shape_m.append(len(matrix[0]))
            matrix = matrix[0]
    return shape_m


def add_matrices2D(mat1, mat2):
    '''Function to add two matrix by element wise'''
    if matrix_shape(mat1) == matrix_shape(mat2):
        addi2mat = [[mat1[i][j] + mat2[i][j] for j in range(
            len(mat1))] for i in range(len(mat2))]
        return addi2mat
