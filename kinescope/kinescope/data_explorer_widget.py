from PyQt5.QtCore import pyqtSignal, QObject, QItemSelection
from PyQt5.QtGui import QIcon, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (QApplication, QAction, QDesktopWidget,
                             QDockWidget, QTreeView, QWidget)


class DataExplorerWidget(QDockWidget):

    new_selection = pyqtSignal(QItemSelection)

    def __init__(self, title, window):
        super(DataExplorerWidget, self).__init__(title, window)
        self.dataView = QTreeView()
        self.dataView.setRootIsDecorated(False)
        self.dataView.setAlternatingRowColors(True)
        # self.widget = QWidget(self)
        self.setWidget(self.dataView)
        self.dataView.setMinimumHeight(200)

        selectionChanged = self.dataView.selectionChanged
        def selection_changed(*args, **kwargs):
            print('new_selection')
            self.new_selection.emit(args[0])
            selectionChanged(*args, **kwargs)

        self.dataView.selectionChanged = selection_changed

    def setModel(self, model):
        self.dataView.setModel(model)
