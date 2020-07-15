#!/usr/bin/env python3
'''Adds two matrices'''


def matrix_shape(matrix):
    '''Matrix shape function'''
    shape_m = []
    shape_m.append(len(matrix))
    if type(matrix[0]) == list:
        while type(matrix[0]) != int:
            shape_m.append(len(matrix[0]))
            matrix = matrix[0]
    return shape_m


def add_matrices(mat1, mat2):
    '''Adding matrices function'''
    if(type(mat1[0]) == int) and (len(mat1) == len(mat2)):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    if(matrix_shape(mat1) == matrix_shape(mat2)):
        if len(matrix_shape(mat1)) == 4:
            return([[[[mat1[i][j][k][l] + mat2[i][j][k][l]
                       for l in range(len(mat1[i][j][k]))]
                      for k in range(len(mat1[i][j]))]
                     for j in range(len(mat1[i]))]
                    for i in range(len(mat1))])

        elif len(matrix_shape(mat1)) == 3:
            return([[[mat1[i][j][k] + mat2[i][j][k]
                      for k in range(len(mat1[i][j]))]
                     for j in range(len(mat1[i]))]
                    for i in range(len(mat1))])
        elif(len(matrix_shape(mat1)) == 2):
            return([[mat1[i][j] + mat2[i][j]
                     for j in range(len(mat1[i]))]
                    for i in range(len(mat1))])
