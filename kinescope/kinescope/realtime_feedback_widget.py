import random

from PyQt5.QtCore import pyqtSignal, Qt, QObject
from PyQt5.QtGui import QIcon, QImage, QPalette, QPixmap, QStandardItemModel
from PyQt5.QtWidgets import (QApplication, QAction, QDesktopWidget, QDockWidget,
    QFileDialog,  QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMenu, QPushButton, QSizePolicy, QScrollArea,
    QSpinBox, QTreeView, QWidget, QVBoxLayout)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class RealtimeFeedbackWidget(QDockWidget):
    def __init__(self, title, window, max_length=100):
        super(QDockWidget, self).__init__(title, window)


        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)

        self.setWidget(self.scroll_area)

        self.data = []
        self.max_length = max_length

    # def setImage(self, qimage):
    #     pixmap = QPixmap.fromImage(qimage)
    #     self.label.setPixmap(pixmap)
    #     self.qimage = qimage
    #     self.label.resize(self.qimage.size())

    def consume(self, joints):
        ''' plot some random stuff '''

        print('RealtimeFeedbackWidget.consume')
        print(joints)

        # random data
        self.data.extend(joints)
        if len(self.data) > self.max_length:
            self.data.pop(0)

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.clear()

        # plot data
        ax.plot(self.data)
        ax.set_ylim(ymin=0, ymax=1.1)
        ax.legend(['x', 'y', 'conf'])

        # refresh canvas
        self.canvas.draw()
