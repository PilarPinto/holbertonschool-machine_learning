#!/usr/bin/env python3
'''Concatenate two matrices along specific axis'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''Using the python concatenate and matrix traverse'''
    cp_mat1 = [ele[:] for ele in mat1]
    cp_mat2 = [ele[:] for ele in mat2]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        new_mat1 = []
        new_mat1 = cp_mat1[:] + cp_mat2[:]
        return new_mat1
    if axis == 1 and (len(mat1) == len(mat2)):
        new_mat2 = []
        for i in range(len(cp_mat2)):
            new_mat2 += [cp_mat1[i] + cp_mat2[i]]
        return new_mat2
    return None
