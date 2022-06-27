import numpy as np
import numpy2qimage
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QImage

from realtime_pipeline import Buffer, Indexer, Pairer, Stage, StatelessStage

class DataStorer(QObject, Stage):

    update_clip = pyqtSignal(int)

    def __init__(self, data_store):
        super(DataStorer, self).__init__()
        self.data_store = data_store
        self.clip_index = None
        self.update_clip.connect(self.data_store.updateClipView)

    def startClip(self, clip_index):
        self.clip_index = clip_index

    def stopClip(self):
        self.clip_index = None

    def consume(self, input_data):
        # BUG (if stopClip() is called from another thread while this thread is inside if-block)
        if self.clip_index is not None:
            self.data_store.addDataToClip(self.clip_index, input_data)
            self.update_clip.emit(self.clip_index)

class FrameToNumpyImage(StatelessStage):
    def __init__(self):
        super(FrameToNumpyImage, self).__init__(lambda frame, emit: emit(frame.image))

class NumpyImageToQImageSignal(QObject, Stage):

    qimage = pyqtSignal(QImage)

    def __init__(self):
        super(QObject, self).__init__()
        super(Stage, self).__init__()

    def consume(self, numpy_image):
        assert(type(numpy_image) == np.ndarray)
        qimage = numpy2qimage.numpy2qimage(numpy_image)
        self.qimage.emit(qimage.copy())
