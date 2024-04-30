#!/usr/bin/env python3

# 21e65bb3-23db-11ec-986f-f39926f24a9c
# e4553c7b-e907-46c6-98e5-f08d1ce8f040

import itertools
import argparse
from typing import Callable
import unittest

import numpy as np

# Bounding boxes and anchors are expected to be Numpy tensors,
# where the last dimension has size 4.

# For bounding boxes in pixel coordinates, the 4 values correspond to:
TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3

def generate_anchors(N: int, image_res: int, scales: list = [1], ratios: list = [1]) -> list:
    """Generate N achors based on scales and ratios lists
    
    Attributes:
    N (int): number of anchors to generate (must be square number)
    scales (list): list of scales to generate anchors
    ratios (list): list of ratios to generate anchors

    Returns:
    np.array: list of anchors in format [top, left, bottom, right]
    int: number of anchors generated
    """

    # check if N is a square number
    assert (N**0.5).is_integer()
    N = int(N**0.5)

    # calculate anchor size based on how many we want in the images
    anchor_size = image_res // N
    anchors = []

    for row, col, scale, ratio in itertools.product(range(N), range(N), scales, ratios):

        # calculate anchor center based on row and col
        center_x, center_y = (col + 0.5) * anchor_size, (row + 0.5) * anchor_size

        # calculate anchor width and height based on scale and ratio
        width = 64
        height = 128

        # calculate anchor top and left corner
        top, left = center_y - height/2, center_x - width/2

        # caluclate anchor bottom and right corner
        bottom, right = center_y + height/2, center_x + width/2

        anchors.append([top, left, bottom, right])

    return np.array(anchors), N**2 * len(scales) * len(ratios)

def clip_bboxes(bboxes: np.ndarray, image_res: int) -> np.ndarray:
    """Clip all given bounding boxes to fit within the image size."""
    return np.clip(bboxes, 0, image_res)

def bboxes_area(bboxes: np.ndarray) -> np.ndarray:
    """ Compute area of given set of bboxes.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    If the bboxes.shape is [..., 4], the output shape is bboxes.shape[:-1].
    """
    return np.maximum(bboxes[:, BOTTOM] - bboxes[:, TOP], 0) \
        * np.maximum(bboxes[:, RIGHT] - bboxes[:, LEFT], 0)

bboxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [0, 0, 20, 20], [5, 5, 10, 10]], np.float32)
assert np.all(bboxes_area(bboxes) == [100, 100, 400, 25])

def bboxes_iou(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """ Compute IoU of corresponding pairs from two sets of bboxes `xs` and `ys`.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    Note that broadcasting is supported, so passing inputs with
    `xs.shape=[num_xs, 1, 4]` and `ys.shape=[1, num_ys, 4]` produces an output
    with shape `[num_xs, num_ys]`, computing IoU for all pairs of bboxes from
    `xs` and `ys`. Formally, the output shape is `np.broadcast(xs, ys).shape[:-1]`.
    """

    intersections = np.stack([
        np.maximum(xs[:, TOP], ys[:, TOP]),
        np.maximum(xs[:, LEFT], ys[:, LEFT]),
        np.minimum(xs[:, BOTTOM], ys[:, BOTTOM]),
        np.minimum(xs[:, RIGHT], ys[:, RIGHT]),
    ], axis=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)

#bboxes1 = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 4, 4], [0, 0, 4, 4]], np.float32)
#bboxes2 = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [2, 0, 4, 4], [0, 0, 8, 8]], np.float32)
#assert np.allclose(bboxes_iou(bboxes1, bboxes2), [1, 0, 0.5, 0.25])

def bboxes_to_rcnn(anchors: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """ Convert `bboxes` to a R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use the TOP, LEFT, BOTTOM, RIGHT constants as indices of the
    respective coordinates.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)

    If the `anchors.shape` is `[anchors_len, 4]` and `bboxes.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """

    # get bbox and anchor centers (y, x)
    bbox_centers = np.array([bboxes[:, TOP] + bboxes[:, BOTTOM], bboxes[:, LEFT] + bboxes[:, RIGHT]]) / 2
    anchor_centers = np.array([anchors[:, TOP] + anchors[:, BOTTOM], anchors[:, LEFT] + anchors[:, RIGHT]]) / 2

    # bbox sizes
    bbox_heights = np.maximum(bboxes[:, BOTTOM] - bboxes[:, TOP], 0)
    bbox_widths = np.maximum(bboxes[:, RIGHT] - bboxes[:, LEFT], 0)

    # anchor sizes
    anchor_heights = np.maximum(anchors[:, BOTTOM] - anchors[:, TOP], 0)
    anchor_widths = np.maximum(anchors[:, RIGHT] - anchors[:, LEFT], 0)

    return np.stack([
        (bbox_centers[0] - anchor_centers[0]) / anchor_heights,
        (bbox_centers[1] - anchor_centers[1]) / anchor_widths,
        np.log(bbox_heights / anchor_heights),
        np.log(bbox_widths / anchor_widths),
    ], axis=-1)

def bboxes_from_rcnn(anchors: np.ndarray, rcnns: np.ndarray) -> np.ndarray:
    """ Convert R-CNN-like representation relative to `anchor` to a `bbox`.

    If the `anchors.shape` is `[anchors_len, 4]` and `rcnns.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """

    # get anchor centers (y, x) and sizes
    anchor_centers = np.array([anchors[:, TOP] + anchors[:, BOTTOM], anchors[:, LEFT] + anchors[:, RIGHT]]) / 2
    anchor_heights = np.maximum(anchors[:, BOTTOM] - anchors[:, TOP], 0)
    anchor_widths = np.maximum(anchors[:, RIGHT] - anchors[:, LEFT], 0)

    # calculate bbox centers, heights and widths
    bbox_centers = np.array([
        rcnns[:, 0] * anchor_heights + anchor_centers[0],
        rcnns[:, 1] * anchor_widths + anchor_centers[1]
    ])
    bbox_heights = np.exp(rcnns[:, 2]) * anchor_heights
    bbox_widths = np.exp(rcnns[:, 3]) * anchor_widths

    # reconstruct bboxes from centers, heights and widths
    return np.stack([
        bbox_centers[0] - bbox_heights / 2,
        bbox_centers[1] - bbox_widths / 2,
        bbox_centers[0] + bbox_heights / 2,
        bbox_centers[1] + bbox_widths / 2
    ], axis=-1)


def bboxes_training(
    anchors: np.ndarray, gold_classes: np.ndarray, gold_bboxes: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` is assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of R-CNN; zeros if no gold object
      was assigned to the anchor
    If the `anchors` shape is `[anchors_len, 4]`, the `anchor_classes` shape
    is `[anchors_len]` and the `anchor_bboxes` shape is `[anchors_len, 4]`.

    Algorithm:
    - First, for each gold object, assign it to an anchor with the largest IoU
      (the one with smaller index if there are several). In case several gold
      objects are assigned to a single anchor, use the gold object with smaller
      index.
    - For each unused anchor, find the gold object with the largest IoU
      (again the one with smaller index if there are several), and if the IoU
      is >= iou_threshold, assign the object to the anchor.
    """

    anchor_classes = np.zeros(anchors.shape[0], np.float32)
    anchor_bboxes = np.zeros((anchors.shape[0], 4), np.float32)

    # print('gold classes:', gold_classes)
    # print('gold bboxes:', gold_bboxes)
    # print('iou threshold:', iou_threshold)

    # TODO: First, for each gold object, assign it to an anchor with the
    # largest IoU (the one with smaller index if there are several). In case
    # several gold objects are assigned to a single anchor, use the gold object
    # with smaller index.

    # reverse gold_classes and gold_bboxes to get the first gold object with the smallest index
    gold_classes, gold_bboxes = gold_classes[::-1],  gold_bboxes[::-1]

    for gold_class, gold_bbox in zip(gold_classes, gold_bboxes):

        # pair gold_bbox to anchor based on highest IoU
        ious = bboxes_iou(anchors, np.array([gold_bbox]))

        # continue if all ious are 0
        if np.all(ious == 0): continue

        # get anchor with highest IoU
        anchor_index = np.argmax(ious)

        # assign gold class to anchor
        anchor_classes[anchor_index] = gold_class + 1

        # assign gold bbox to anchor
        anchor_bboxes[anchor_index] = bboxes_to_rcnn(np.array([anchors[anchor_index]]), np.array([gold_bbox]))[0]

    
    # TODO: For each unused anchor, find the gold object with the largest IoU
    # (again the one with smaller index if there are several), and if the IoU
    # is >= threshold, assign the object to the anchor.

    for anchor_idx, unused_anchor in enumerate(anchor_classes):
        if unused_anchor != 0: continue

        # get IoUs of anchor with gold bboxes
        ious = bboxes_iou(np.array([anchors[anchor_idx]]), gold_bboxes)

        # skip loop if all ious are 0
        if np.sum(ious >= iou_threshold) == 0: continue

        # get gold object with highest IoU and assign it to anchor
        gold_index = np.argmax(ious)
        anchor_classes[anchor_idx] = gold_classes[gold_index] + 1

        # assign gold bbox to anchor
        anchor_bboxes[anchor_idx] = bboxes_to_rcnn(np.array([anchors[anchor_idx]]), np.array([gold_bboxes[gold_index]]))[0]


    return anchor_classes, anchor_bboxes

def main(args: argparse.Namespace) -> tuple[Callable, Callable, Callable]:

    return bboxes_to_rcnn, bboxes_from_rcnn, bboxes_training


class Tests(unittest.TestCase):
    def test_bboxes_to_from_rcnn(self):
        data = [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 30], [.5, 1, np.log(2), np.log(3)]],
            [[0, 9, 10, 19], [2, 10, 5, 16], [-0.15, -0.1, -1.20397, -0.51083]],
            [[5, 3, 15, 13], [7, 7, 10, 9], [-0.15, 0, -1.20397, -1.60944]],
            [[7, 6, 17, 16], [9, 10, 12, 13], [-0.15, 0.05, -1.20397, -1.20397]],
            [[5, 6, 15, 16], [7, 7, 10, 10], [-0.15, -0.25, -1.20397, -1.20397]],
            [[6, 3, 16, 13], [8, 5, 12, 8], [-0.1, -0.15, -0.91629, -1.20397]],
            [[5, 2, 15, 12], [9, 6, 12, 8], [0.05, 0, -1.20397, -1.60944]],
            [[2, 10, 12, 20], [6, 11, 8, 17], [0, -0.1, -1.60944, -0.51083]],
            [[10, 9, 20, 19], [12, 13, 17, 16], [-0.05, 0.05, -0.69315, -1.20397]],
            [[6, 7, 16, 17], [10, 11, 12, 14], [0, 0.05, -1.60944, -1.20397]],
            [[2, 2, 12, 12], [3, 5, 8, 8], [-0.15, -0.05, -0.69315, -1.20397]],
        ]
        # First run on individual anchors, and then on all together
        for anchors, bboxes, rcnns in [map(lambda x: [x], row) for row in data] + [zip(*data)]:
            anchors, bboxes, rcnns = [np.array(data, np.float32) for data in [anchors, bboxes, rcnns]]
            np.testing.assert_almost_equal(bboxes_to_rcnn(anchors, bboxes), rcnns, decimal=3)
            np.testing.assert_almost_equal(bboxes_from_rcnn(anchors, rcnns), bboxes, decimal=3)

    def test_bboxes_training(self):
        anchors = np.array([[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]], np.float32)
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
                [[1], [[14., 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(.2), np.log(.2)]], 0.5],
                [[2], [[0., 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
                [[2], [[0., 0, 20, 20]], [3, 3, 3, 3],
                 [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 0, 1],
                 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-0.35, -0.45, 0.53062, 0.40546]], 0.5],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 2, 1],
                 [[0, 0, 0, 0], [0, 0, 0, 0], [-0.1, 0.6, -0.22314, 0.69314], [-0.35, -0.45, 0.53062, 0.40546]], 0.3],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 1, 2, 1],
                 [[0, 0, 0, 0], [0.65, -0.45, 0.53062, 0.40546], [-0.1, 0.6, -0.22314, 0.69314],
                  [-0.35, -0.45, 0.53062, 0.40546]], 0.17],
        ]:
            gold_classes, anchor_classes = np.array(gold_classes, np.int32), np.array(anchor_classes, np.int32)
            gold_bboxes, anchor_bboxes = np.array(gold_bboxes, np.float32), np.array(anchor_bboxes, np.float32)
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            np.testing.assert_almost_equal(computed_classes, anchor_classes, decimal=3)
            np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)


if __name__ == '__main__':
    unittest.main()

