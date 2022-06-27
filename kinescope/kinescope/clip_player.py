import os

from collections import namedtuple
import threading
import time

import imageio
import numpy as np

from realtime_pipeline import Stage


class ClipPlayer(Stage):

    def __init__(self, data_store):
        super(ClipPlayer, self).__init__()

        self.data_store = data_store

        self.clip_path = None
        self.reader = None
        self.frames_played = 0
        self.clip_length = 0

        self.fps = 30

        self.thread = threading.Thread(target=self.worker, args=())
        self.thread.daemon = True
        self.thread.start()

    def worker(self):

        while True:

            if self.is_playing():
                print('self.frames_played = {}'.format(self.frames_played))

                data = self.reader.get_next_data()
                image = np.array(data)
                self.emit(image)

                self.frames_played += 1

            time.sleep(1 / self.fps)

    def is_playing(self):
        return self.frames_played < self.clip_length

    def play(self, clip_path):
        self.clip_path = clip_path
        self.reader = imageio.get_reader(clip_path)
        self.clip_length = self.reader.count_frames()
        self.frames_played = 0

    def handleNewSelection(self, q_item_selection):
        indexes = q_item_selection.indexes()
        if len(indexes) > 0:
            first_index = indexes[0]
            clip_idx = first_index.row()

            print('playing clip {}'.format(clip_idx))

            clip_uv_dir = self.data_store.clips[clip_idx].output_dir_uv
            video_path = os.path.join(clip_uv_dir, 'video.mp4')
            self.play(video_path)
