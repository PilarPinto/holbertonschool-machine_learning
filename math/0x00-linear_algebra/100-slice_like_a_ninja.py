#!/usr/bin/env python3
'''Script to slice'''


def np_slice(matrix, axes={}):
    '''Use dinamic dictionaty to use the info  and
    creates a cast of tuples from slides'''

    maxAxis = max(axes.keys())
    dimension_axis = {}

    for key in range(maxAxis):
        dimension_axis[key] = (None, None, None)
    for slide_key in axes.keys():
        dimension_axis[slide_key] = axes[slide_key]

    return (matrix[tuple(slice(*dimension_axis[x]) for x in dimension_axis)])
