from wink.writers import ThreadedWriter, ProcessWriter
from wink.pipeline import Stage


class ThreadedWriterStage(Stage):

    def __init__(self, *args, **kwargs):
        self.writer = ThreadedWriter(*args, **kwargs)

    def consume(self, frame):
        self.writer.append_data(frame.image)

    def is_done(self):
        return self.writer.is_done()

    def close(self):
        self.writer.close()


class ProcessWriterStage(Stage):

    def __init__(self, *args, **kwargs):
        self.writer = ProcessWriter(*args, **kwargs)

    def consume(self, frame):
        self.writer.append_data(frame.image)

    def is_done(self):
        return self.writer.is_done()

    def close(self):
        self.writer.close()
