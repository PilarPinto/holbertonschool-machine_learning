#!/usr/bin/env python3
'''
Script two add two matix by elements-wise
'''


def add_matrices2D(mat1, mat2):
    '''Function to add two matrix by element wise'''
    if (len(mat1) == len(mat2)) and (len(mat1[0]) == len(mat2[0])):
        addi2mat = [[mat1[i][j] + mat2[i][j] for j in range(
            len(mat1[0]))] for i in range(len(mat1))]
        return addi2mat
