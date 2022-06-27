from collections import namedtuple
import threading
import time

import numpy as np
from skimage.io import imread

from realtime_pipeline import Stage

Frame = namedtuple('Frame', ['timestamp', 'image'])

class StaticCamera(Stage):

    def __init__(self, image, fps, alternating=False):
        super(StaticCamera, self).__init__()

        if type(image) is str:
            image = imread(image)

        self.image = image
        self.fps = fps
        self.alternating = alternating
        self.state = False
        self.is_grabbing = False

        self.start_thread()

    def start_thread(self):
        t = threading.Thread(target=self.worker, args=())
        self.thread = t
        t.daemon = True
        t.start()

    def start_grabbing(self):
        self.is_grabbing = True

    def stop_grabbing(self):
        self.is_grabbing = False

    def worker(self):
        while True:
            if self.is_grabbing:
                frame = self.fake_frame()
                self.emit(frame)
            time.sleep(1 / self.fps)

    def fake_frame(self):
        image = self.image
        if self.alternating:
            if self.state:
                image = image * 3
            self.state = not self.state
        timestamp = time.time()
        frame = Frame(image=image, timestamp=timestamp)
        return frame
