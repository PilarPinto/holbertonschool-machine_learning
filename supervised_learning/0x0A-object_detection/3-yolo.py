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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        '''Filter boxes over the threshold'''
        scores = []

        for i in range(len(boxes)):
            scores.append(box_confidences[i] * box_class_probs[i])

        filter_boxes = [box.reshape(-1, 4) for box in boxes]
        filter_boxes = np.concatenate(filter_boxes)
        class_m = [np.argmax(box, -1) for box in scores]
        class_m = [box.reshape(-1) for box in class_m]
        class_m = np.concatenate(class_m)

        class_scores = [np.max(box, -1) for box in scores]
        class_scores = [box.reshape(-1) for box in class_scores]
        class_scores = np.concatenate(class_scores)

        f_mask = np.where(class_scores >= self.class_t)
        filtered_boxes = filter_boxes[f_mask]
        box_classes = class_m[f_mask]
        box_scores = class_scores[f_mask]

        return(filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        '''non max suppression'''

        box_pred = []
        box_classes_pred = []
        box_scores_pred = []
        u_classes = np.unique(box_classes)
        for ucls in u_classes:
            idx = np.where(box_classes == ucls)

            bfilters = filtered_boxes[idx]
            bscores = box_scores[idx]
            bclasses = box_classes[idx]

            pick = self._intersectionou(bfilters, self.nms_t, bscores)

            filters = bfilters[pick]
            scores = bscores[pick]
            classes = bclasses[pick]

            box_pred.append(filters)
            box_classes_pred.append(classes)
            box_scores_pred.append(scores)
        filtered_boxes = np.concatenate(box_pred, axis=0)
        box_classes = np.concatenate(box_classes_pred, axis=0)
        box_scores = np.concatenate(box_scores_pred, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def _intersectionou(self, filtered_boxes, thresh, scores):
        '''Compute intersection'''
        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = scores.argsort()[::-1]

        pick = []
        while idxs.size > 0:
            i = idxs[0]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            inter = (w * h)
            overlap = inter / (area[i] + area[idxs[1:]] - inter)
            ind = np.where(overlap <= self.nms_t)[0]
            idxs = idxs[ind + 1]

        return pick
