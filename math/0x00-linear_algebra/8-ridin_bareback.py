#!/usr/bin/env python3
'''Matrix multiplication'''


def mat_mul(mat1, mat2):
    '''Matrix multiplication'''
    multiply = [[0 for cols in range(len(mat2[0]))]
                for rows in range(len(mat1))]
    if(range(len(mat1[0])) == range(len(mat2))):
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                for k in range(len(mat2)):
                    multiply[i][j] += mat1[i][k] * mat2[k][j]
    else:
        return(None)
    return multiply
