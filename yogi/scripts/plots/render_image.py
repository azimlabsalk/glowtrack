#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt

from yogi.db import *
from yogi.models import *


def render_image(image_id, source_id, output_path):

    image = session.query(Image).filter_by(id=image_id).one()
    label = session.query(Label).filter_by(image_id=image_id, source_id=source_id).one()

    arr = image.get_array()
    plt.imshow(arr)

    plt.scatter(label.x_px, label.y_px)

    plt.axis('off')

    plt.savefig(output_path)


if __name__ == '__main__':

    image_id = int(sys.argv[1])
    source_id = int(sys.argv[2])
    output_path = sys.argv[3]

    render_image(image_id, source_id, output_path)

