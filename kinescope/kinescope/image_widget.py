from PyQt5.QtCore import pyqtSignal, Qt, QObject
from PyQt5.QtGui import QIcon, QImage, QPalette, QPixmap, QStandardItemModel
from PyQt5.QtWidgets import (QApplication, QAction, QDesktopWidget, QDockWidget,
    QFileDialog,  QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMenu, QPushButton, QSizePolicy, QScrollArea,
    QSpinBox, QTreeView, QWidget, QVBoxLayout)


class ImageWidget(QDockWidget):
    def __init__(self, title, window):
        super(QDockWidget, self).__init__(title, window)

        self.qimage = None

        self.scroll_area = QScrollArea()

        self.label = QLabel()
        self.label.setBackgroundRole(QPalette.Dark)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setScaledContents(True)

        self.scroll_area.setWidget(self.label)

        self.setWidget(self.scroll_area)

    def setImage(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)
        self.qimage = qimage
        self.label.resize(self.qimage.size())

