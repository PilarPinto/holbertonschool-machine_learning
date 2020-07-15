#!/usr/bin/env python3
'''
Script with adding element-wise
'''


def add_arrays(arr1, arr2):
    '''Traverse and sum each component by x y z'''
    add_arr = []
    if len(arr1) != len(arr2):
        return(None)
    for index in range(0, len(arr1)):
        add_arr.append(arr1[index] + arr2[index])
        if arr1[index] == 0:
            arr.pop(index)
    return add_arr
