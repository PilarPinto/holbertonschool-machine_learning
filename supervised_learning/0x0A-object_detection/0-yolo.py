#!/usr/bin/env python3
'''
algorithm to perform object detection
File in YOLO dataset
'''
import tensorflow.keras as K


class Yolo:
    '''
    Initialize the class YOLO
    '''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''uses the Yolo v3 algorithm to perform object detection'''
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
