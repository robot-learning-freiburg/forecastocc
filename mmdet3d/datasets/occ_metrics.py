# Copyright (c) OpenMMLab. All rights reserved.
"""Occupancy forecasting metrics."""

import numpy as np


OCCUPANCY_CLASS_NAMES = (
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free')


class OccupancyMetrics:
    """Accumulate full-volume semantic and occupied/free occupancy metrics.

    Confusion-matrix rows are ground-truth classes and columns are predicted
    classes. All values returned by :meth:`compute` are scalar percentages.
    """

    def __init__(self, class_names=OCCUPANCY_CLASS_NAMES):
        self.class_names = tuple(class_names)
        if len(self.class_names) < 2:
            raise ValueError('At least one semantic class and free are needed.')

        self.num_classes = len(self.class_names)
        self.free_class = self.num_classes - 1
        self.confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64)
        self.sample_count = 0

    def add_batch(self, prediction, target):
        """Add one sample without modifying any caller-owned arrays."""
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError(
                f'Prediction shape {prediction.shape} does not match target '
                f'shape {target.shape}.')

        valid = (target >= 0) & (target < self.num_classes)
        selected_prediction = prediction[valid]
        selected_target = target[valid]
        if np.any((selected_prediction < 0) |
                  (selected_prediction >= self.num_classes)):
            raise ValueError('Predictions contain an invalid class index.')

        encoded = (self.num_classes * selected_target.astype(np.int64) +
                   selected_prediction.astype(np.int64))
        self.confusion += np.bincount(
            encoded, minlength=self.num_classes**2).reshape(
                self.num_classes, self.num_classes)
        self.sample_count += 1

    @staticmethod
    def _percentage(numerator, denominator):
        if denominator == 0:
            return 0.0
        return float(numerator / denominator * 100.0)

    def compute(self, prefix=''):
        """Return scalar semantic and occupied/free metrics in percent."""
        key_prefix = f'{prefix}_' if prefix else ''
        diagonal = np.diag(self.confusion).astype(np.float64)
        union = (self.confusion.sum(axis=1) +
                 self.confusion.sum(axis=0) - diagonal)

        # Keep the established ForecastOcc convention: a class with no union
        # contributes zero rather than being omitted from semantic mIoU.
        class_iou = diagonal / (union + 1e-6) * 100.0
        semantic_iou = class_iou[:self.free_class]

        true_positive = self.confusion[:self.free_class,
                                       :self.free_class].sum()
        false_positive = self.confusion[self.free_class,
                                        :self.free_class].sum()
        false_negative = self.confusion[:self.free_class,
                                        self.free_class].sum()

        metrics = {
            f'{key_prefix}miou': float(np.mean(semantic_iou)),
            f'{key_prefix}iou': self._percentage(
                true_positive,
                true_positive + false_positive + false_negative),
            f'{key_prefix}precision': self._percentage(
                true_positive, true_positive + false_positive),
            f'{key_prefix}recall': self._percentage(
                true_positive, true_positive + false_negative),
        }
        metrics.update({
            f'{key_prefix}iou_{class_name}': float(class_iou[index])
            for index, class_name in enumerate(
                self.class_names[:self.free_class])
        })
        return metrics
