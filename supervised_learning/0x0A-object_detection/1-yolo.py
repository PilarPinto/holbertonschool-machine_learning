#!/usr/bin/env python3
'''
algorithm to perform object detection
File in YOLO dataset
'''
import tensorflow.keras as K
import numpy as np


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

    def sigmoid(self, x):
        '''Sigmoid function'''
        return(1/(1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        '''Find and return the tuple of outputs in the detection process'''
        boxes, box_confidences, box_class_probs = [], [], []
        img_h, img_w = image_size
        for ind in range(len(outputs)):
            # input sizes
            input_weight = self.model.input_shape[1]
            input_height = self.model.input_shape[2]

            grid_he = outputs[ind].shape[0]
            grid_we = outputs[ind].shape[1]
            anchor_boxes = outputs[ind].shape[2]

            tx = outputs[ind][..., 0]
            ty = outputs[ind][..., 1]
            tw = outputs[ind][..., 2]
            th = outputs[ind][..., 3]

            c = np.zeros((grid_he, grid_we, anchor_boxes))
            idx_y = np.arange(grid_he)
            idx_y = idx_y.reshape(grid_he, 1, 1)
            idx_x = np.arange(grid_we)
            idx_x = idx_x.reshape(1, grid_we, 1)
            cx = c + idx_x
            cy = c + idx_y

            p_w = self.anchors[ind, :, 0]
            p_h = self.anchors[ind, :, 1]

            # Using cx (width) cy (height) for cbounding
            b_x = self.sigmoid(tx) + cx
            b_y = self.sigmoid(ty) + cy
            b_w = p_w * np.exp(tw)
            b_h = p_h * np.exp(th)

            # normalizing
            bx = b_x / grid_we
            by = b_y / grid_he
            bw = b_w / input_weight
            bh = b_h / input_height

            bx1 = bx - bw / 2
            by1 = by - bh / 2
            bx2 = bx + bw / 2
            by2 = by + bh / 2

            outputs[ind][..., 0] = bx1 * img_w
            outputs[ind][..., 1] = by1 * img_h
            outputs[ind][..., 2] = bx2 * img_w
            outputs[ind][..., 3] = by2 * img_h

            boxes.append(outputs[ind][..., 0:4])
            box_confidences.append(self.sigmoid(outputs[ind][..., 4:5]))
            box_class_probs.append(self.sigmoid(outputs[ind][..., 5:]))
        return(boxes, box_confidences, box_class_probs)
