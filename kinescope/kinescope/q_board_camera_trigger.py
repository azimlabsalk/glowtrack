from PyQt5.QtCore import pyqtSignal, QObject

from board_camera_trigger import BoardCameraTrigger


class QBoardCameraTrigger(BoardCameraTrigger, QObject):

    def __init__(self, port=None):
        QObject.__init__(self)
        BoardCameraTrigger.__init__(self, port=port)

