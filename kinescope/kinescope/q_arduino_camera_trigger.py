from PyQt5.QtCore import pyqtSignal, QObject

from arduino_camera_trigger import ArduinoCameraTrigger


class QArduinoCameraTrigger(ArduinoCameraTrigger, QObject):

    def __init__(self, port=None):
        QObject.__init__(self)
        ArduinoCameraTrigger.__init__(self, port=port)

