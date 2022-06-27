from collections import namedtuple

import numpy as np

from realtime_pipeline import Stage


Frame = namedtuple('Frame', ['timestamp', 'image'])

class Flipper(Stage):

    def __init__(self):
        super(Flipper, self).__init__()
        self.flip_ud = False
        self.flip_lr = False

    def set_flip_ud(self, flag):
        self.flip_ud = flag

    def set_flip_lr(self, flag):
        self.flip_lr = flag

    def consume(self, input_frame):
        image = input_frame.image

        if self.flip_ud:
            image = np.flipud(image)

        if self.flip_lr:
            image = np.fliplr(image)

        output_frame = Frame(timestamp=input_frame.timestamp, image=image)
        self.emit(output_frame)

