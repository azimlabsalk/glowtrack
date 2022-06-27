"""Code for dealing with manual annotation data."""

import numpy as np

from yogi.models import Label


def merge_labels(labels1, labels2, threshold):
    merged_labels = [merge_label(label1, label2) for (label1, label2) in
                     zip(labels1, labels2) if
                     both_hidden(label1, label2) or
                     labels_match(label1, label2, threshold)]

    return merged_labels


def merge_label(label1, label2):
    assert(label1.image_id == label2.image_id)
    assert(label1.landmark_id == label2.landmark_id)

    x = None
    y = None

    if not both_hidden(label1, label2):
        x = np.mean([label1.x, label2.x])
        y = np.mean([label1.y, label2.y])

    occluded = label1.occluded or label2.occluded

    image_id = label1.image_id
    landmark_id = label1.landmark_id

    new_label = Label(image_id=image_id, x=x, y=y,
                      occluded=occluded, landmark_id=landmark_id)

    return new_label


def both_hidden(label1, label2):
    return label1.is_hidden() and label2.is_hidden()


def labels_match(label1, label2, threshold):
    return not label1.is_hidden() and not label2.is_hidden() and pixel_distance(label1, label2) < threshold


def pixel_distance(label1, label2):
    diff_x = label1.x_px - label2.x_px
    diff_y = label1.y_px - label2.y_px
    distance = np.linalg.norm([diff_x, diff_y])
    return distance

