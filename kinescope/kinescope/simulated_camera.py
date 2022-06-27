from collections import namedtuple
import threading
import time

import numpy as np

from realtime_pipeline import Stage

Frame = namedtuple('Frame', ['timestamp', 'image'])


class SimulatedCamera(Stage):

    def __init__(self, width, height, fps, alternating=False):
        super(SimulatedCamera, self).__init__()

        self.is_grabbing = False

        self.width = width
        self.height = height
        self.fps = fps
        self.alternating = alternating
        self.state = False

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
        image = self.fake_image()
        if self.alternating:
            if self.state:
                image = image * 3
            self.state = not self.state
        timestamp = time.time()
        frame = Frame(image=image, timestamp=timestamp)
        return frame

    def fake_image(self):
        return random_image(self.width, self.height)


def random_image(width, height):
    image = np.random.randint(255, size=(height, width))
    return image
