import os
from queue import Queue
import threading

import imageio
import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt, QObject
from PyQt5.QtGui import QIcon, QImage, QPalette, QPixmap, QStandardItemModel
from skimage.io import imsave

from settings import UV_INDEX, VISIBLE_INDEX

NAME, LENGTH, PROGRESS = range(3)


class FrameWriter(QObject):

    frame_written = pyqtSignal(int)

    def __init__(self, num_workers=1, save_video=True):
        super(FrameWriter, self).__init__()
        self.num_workers = num_workers
        self.workers = []
        self.work_q = Queue()
        self.video_writers = {}
        self.createWorkers()

    def finish_clip(self, clip_index):
        self.work_q.put(None)

    def add(self, data, frame_idx, dir_uv, dir_vis, clip_index):
        self.work_q.put((data, frame_idx, dir_uv, dir_vis, clip_index))

    def createWorkers(self):
        for _ in range(self.num_workers):
            t = threading.Thread(target=self.worker, args=())
            self.workers.append(t)
            t.daemon = True
            t.start()

    def worker(self):
        while True:
            data = self.work_q.get()
            if data is None:
                # release writers
                for writer in self.video_writers.values():
                    writer.close()
                    # writer.release()
                self.video_writers = {}
                self.work_q.task_done()
            else:
                frame, frame_idx, dir_uv, dir_vis, clip_index = data
                frame_visible, frame_uv = frame[VISIBLE_INDEX], frame[UV_INDEX]

                if len(self.video_writers) == 0:
                    fps = 30
                    vis_fname = dir_vis + '/video.mp4'
                    vis_writer = imageio.get_writer(vis_fname, fps=fps,
                                                    quality=10)
                    self.video_writers['visible'] = vis_writer
                    uv_fname = dir_uv + '/video.mp4'
                    uv_writer = imageio.get_writer(uv_fname, fps=fps,
                                                   quality=10)
                    self.video_writers['uv'] = uv_writer

                img = frame_visible.image
                # img = cv2.cvtColor(frame_visible.image, cv2.COLOR_BGR2RGB)
                self.video_writers['visible'].append_data(img)

                img = frame_uv.image
                # img = cv2.cvtColor(frame_uv.image, cv2.COLOR_BGR2RGB)
                self.video_writers['uv'].append_data(img)

                # if len(self.video_writers) == 0:
                #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                #     size = frame_visible.image.shape[0:2]
                #     fps = 30
                #     vis_fname = dir_vis + '/video.mp4'
                #     vis_writer = cv2.VideoWriter(vis_fname, fourcc, fps, size)
                #     self.video_writers['visible'] = vis_writer
                #     uv_fname = dir_uv + '/video.mp4'
                #     uv_writer = cv2.VideoWriter(uv_fname, fourcc, fps, size)
                #     self.video_writers['uv'] = uv_writer
                #
                # img = cv2.cvtColor(frame_visible.image, cv2.COLOR_BGR2RGB)
                # self.video_writers['visible'].write(img)
                #
                # img = cv2.cvtColor(frame_uv.image, cv2.COLOR_BGR2RGB)
                # self.video_writers['uv'].write(img)

                # save_frame(frame_visible.image, frame_idx, dir_vis)
                # save_frame(frame_uv.image, frame_idx, dir_uv)

                self.frame_written.emit(clip_index)
                self.work_q.task_done()


class Clip(QObject):

    clip_updated = pyqtSignal(int)

    def __init__(self, name, clip_index, project_dir, frame_writer):
        super(Clip, self).__init__()
        self.name = name
        self.clip_index = clip_index
        self.project_dir = project_dir
        self.frame_writer = frame_writer

        self.make_output_dirs()

        self.length = 0
        self.written = 0

    def make_output_dirs(self):
        self.clip_dir = os.path.join(self.project_dir, self.name)
        self.output_dir_visible = os.path.join(self.clip_dir, 'visible')
        self.output_dir_uv = os.path.join(self.clip_dir, 'uv')
        os.makedirs(self.clip_dir)
        os.makedirs(self.output_dir_visible)
        os.makedirs(self.output_dir_uv)

    def addData(self, data):
        """NOTE: This function is not threadsafe as written.
        Frames should be added from a single thread.
        """
        frame_idx = self.length
        self.length += 1
        self.frame_writer.add(data, frame_idx, self.output_dir_uv,
                              self.output_dir_visible, self.clip_index)


def save_frame(image, frame_idx, output_dir):
    fname = '{:08d}.png'.format(frame_idx)
    path = os.path.join(output_dir, fname)
#    imsave(path, image, plugin='imageio', compress_level=3)
    imsave(path, image)


class VideoDataStore(QObject):

    Headers = ['Name', 'Length', 'Frames saved']

    def __init__(self, project_dir, num_workers=1):
        super(VideoDataStore, self).__init__()
        self.project_dir = project_dir
        self.clips = []
        self.model = QStandardItemModel(0, len(self.Headers), None)
        self.writer = FrameWriter(num_workers=num_workers)
        self.writer.frame_written.connect(self.frame_written)
        self.name_template = 'clip_{}'
        self.setHeaderData()

    def set_project_dir(self, project_dir):
        self.project_dir = project_dir

    def setHeaderData(self):
        for i, string in enumerate(self.Headers):
            self.model.setHeaderData(i, Qt.Horizontal, string)

    def itemModel(self):
        return self.model

    def is_busy(self):
        return False

    # clip data methods

    def setClipNameTemplate(self, name_template):
        self.name_template = name_template

    def createClip(self):
        clipIndex = len(self.clips)
        name = self.name_template.format(clipIndex)
        newClip = Clip(name, clipIndex, self.project_dir, self.writer)
        self.clips.append(newClip)
        self.addClipToView(newClip, clipIndex)
        return clipIndex

    def addDataToClip(self, clipIndex, data):
        self.clips[clipIndex].addData(data)

    def clipFinished(self, index):
        self.writer.finish_clip(index)

    def frame_written(self, clip_index):
        self.clips[clip_index].written += 1
        self.updateClipView(clip_index)

    # clip view methods

    def addClipToView(self, newClip, clipIndex):
        self.model.insertRow(clipIndex)
        newClip.clip_updated.connect(self.updateClipView)
        newClip.clip_updated.emit(clipIndex)

    def updateClipView(self, clipIndex):
        clip = self.clips[clipIndex]
        self.model.setData(self.model.index(clipIndex, NAME), clip.name)
        self.model.setData(self.model.index(clipIndex, LENGTH), clip.length)
        self.model.setData(self.model.index(clipIndex, PROGRESS), clip.written)
