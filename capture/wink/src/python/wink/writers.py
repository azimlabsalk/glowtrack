import multiprocessing
import queue
from threading import Thread

import imageio


class ThreadedWriter(object):

    def __init__(self, *args, **kwargs):
        self.writer = imageio.get_writer(*args, **kwargs)
        self.q = queue.Queue()
        self.init_thread()

    def init_thread(self):
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def append_data(self, img):
        self.q.put(img)

    def close(self):
        self.writer.close()

    def is_done(self):
        return self.q.empty()

    def run(self):
        while True:
            img = self.q.get()
            self.writer.append_data(img)
            self.q.task_done()


class ProcessWriter(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.q = multiprocessing.Queue()
        self.init_thread()

    def init_thread(self):
        self.thread = multiprocessing.Process(target=run_writer,
            args=(self.q, self.args, self.kwargs))
        self.thread.daemon = True
        self.thread.start()

    def append_data(self, img):
        size = self.q.qsize()
        self.q.put(img)

    def close(self):
        self.q.put(None)

    def is_done(self):
        return self.q.empty()


def run_writer(q, args, kwargs):
    writer = imageio.get_writer(*args, **kwargs)
    while True:
        img = q.get()
        if img is not None:
            writer.append_data(img)
        else:
            writer.close()
            break
