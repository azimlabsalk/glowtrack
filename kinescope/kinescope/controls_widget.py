from PyQt5.QtCore import pyqtSignal, Qt, QObject
from PyQt5.QtGui import QIcon, QImage, QPalette, QPixmap, QStandardItemModel
from PyQt5.QtWidgets import (QApplication, QAction, QCheckBox, QDesktopWidget,
                             QDockWidget, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QLabel,
                             QLineEdit, QListWidget, QMainWindow, QMenu,
                             QPushButton, QSizePolicy, QScrollArea, QSpinBox,
                             QTreeView, QWidget, QVBoxLayout)


class ControlsWidget(QDockWidget):

    def __init__(self, title, window, include_nn=False):
        super(ControlsWidget, self).__init__(title, window)
        self.widget = QWidget(self)
        self.setWidget(self.widget)

        self.nn_controls = NeuralNetworkControlsBox()
        self.camera_controls = CameraControlsBox()
        self.data_controls = DataControlsBox()
        self.trigger_controls = TriggerControlsBox()
        self.boxes = [self.camera_controls, self.data_controls,
                      self.trigger_controls]
        if include_nn:
            self.boxes.append(self.nn_controls)

        self.groups_layout = QVBoxLayout()
        for box in self.boxes:
            self.groups_layout.addWidget(box)

        self.groups_layout.addStretch(1)

        self.widget.setLayout(self.groups_layout)
        self.widget.setMaximumWidth(300)


class NeuralNetworkControlsBox(QGroupBox):

    loadNet = pyqtSignal(str)

    def __init__(self):
        super(NeuralNetworkControlsBox, self).__init__("Neural Network")

        self.form_layout = QFormLayout()

        self.path_row_layout = QGridLayout()
        self.path_text_widget = QLineEdit()
        self.path_row_layout.addWidget(self.path_text_widget, 0, 0)
        self.browse_button = QPushButton("Browse")
        self.load_button = QPushButton("Load")
        self.path_row_layout.addWidget(self.browse_button, 0, 3)

        self.browse_button.clicked.connect(self.browse)
        self.load_button.clicked.connect(self.load_pressed)
        # self.path_text_widget.editingFinished.connect(self.editingFinished)

        self.conf_row_layout = QFormLayout()
        self.conf_threshold_widget = ConfThresholdWidget()
        self.conf_row_layout.addRow("&Confidence threshold:",
                                    self.conf_threshold_widget)

        self.form_layout.addRow(self.path_row_layout)
        self.form_layout.addRow(self.load_button)
        self.form_layout.addRow(self.conf_row_layout)
        self.setLayout(self.form_layout)

    def browse(self):

        (path, file_type) = QFileDialog.getOpenFileName(self,
            "Load DeeperCut YAML", "~/", "YAML files (*.yaml)")

#        (path, file_type) = QFileDialog.getOpenFileName(self,
#            "Load Neural Network", "~/", "PyTorch files (*.pth)")

        self.set_path(path)

    def editingFinished(self):
        self.set_path(self.path_text_widget.text())

    def set_path(self, path):
        if self.is_valid_nn(path):
            self.path = path
            self.path_text_widget.setText(path)
            self.loadNet.emit(self.path)
        else:
            print('Invalid neural network file')

    def load_pressed(self):
        self.set_path(self.path_text_widget.text())

    def is_valid_nn(self, path):
        return True


class CameraControlsBox(QGroupBox):

    def __init__(self):
        super(CameraControlsBox, self).__init__("Camera")
        self.camera_controls_layout = QFormLayout()
        self.buffer_size_widget = BufferSizeWidget()
        self.camera_controls_layout.addRow("&Buffer size:", self.buffer_size_widget)
        self.skip_frames_widget = SkipFramesWidget()
        self.camera_controls_layout.addRow("&Pairs to skip:", self.skip_frames_widget)
        self.flip_lr_checkbox = QCheckBox(self)
        self.camera_controls_layout.addRow("&Flip left-right:", self.flip_lr_checkbox)
        self.flip_ud_checkbox = QCheckBox(self)
        self.camera_controls_layout.addRow("&Flip up-down:", self.flip_ud_checkbox)
        self.setLayout(self.camera_controls_layout)


class DataControlsBox(QGroupBox):

    def __init__(self):
        super(DataControlsBox, self).__init__("Data")
        self.controls_layout = QFormLayout()
        self.clip_name = QLineEdit()
        self.controls_layout.addRow("&Clip path:", self.clip_name)
        self.setLayout(self.controls_layout)


class TriggerControlsBox(QGroupBox):

    def __init__(self):
        super(TriggerControlsBox, self).__init__("Triggers")
        self.form_layout = QFormLayout()

        self.cycle_duration = TimeSpinBox()
        self.form_layout.addRow("&Cycle duration:", self.cycle_duration)

        self.timeboxes = []
        self.timebox_labels = [
            "&UV light on:",
            "&UV light off:",
            "&UV cam on:",
            "&UV cam off:",
            "&Visible light on:",
            "&Visible light off:",
            "&Visible cam on:",
            "&Visible cam off:",
        ]

        for i in range(8):
            timebox = TimeSpinBox()
            label = self.timebox_labels[i]
            self.form_layout.addRow(label, timebox)
            self.timeboxes.append(timebox)

        self.setLayout(self.form_layout)


class BufferSizeWidget(QSpinBox):

    def __init__(self, maximum=1000):
        super(BufferSizeWidget, self).__init__()
        self.setMaximum(maximum)

    def set_buffer(self, buffer):
        self.buffer = buffer
        self.setValue(self.buffer.max_size)
        self.valueChanged.connect(self.buffer.set_max_size)


class TimeSpinBox(QDoubleSpinBox):

    def __init__(self):
        super(TimeSpinBox, self).__init__()
        self.setMinimum(0.0)
        self.setMaximum(10000.0)
        self.setSingleStep(0.1)
        self.setDecimals(4)

    # def set_time_object(self, time_object):
    #     self.time_object = time_object
    #     self.setValue(self.time_object.time)
    #     self.valueChanged.connect(self.time_object.set_time)


class ConfThresholdWidget(QDoubleSpinBox):

    def __init__(self):
        super(ConfThresholdWidget, self).__init__()
        self.setMaximum(1.0)
        self.setMinimum(0.0)
        self.setSingleStep(0.1)
        self.setDecimals(4)

    def set_thresholder(self, threshold_object):
        self.threshold_object = threshold_object
        self.setValue(self.threshold_object.conf_threshold)
        self.valueChanged.connect(self.threshold_object.set_conf_threshold)


class SkipFramesWidget(QSpinBox):

    def __init__(self, maximum_frames=1000):
        super(SkipFramesWidget, self).__init__()
        self.setMaximum(maximum_frames)

    def set_skipper(self, skipper):
        self.skipper = skipper
        self.setValue(self.skipper.num_frames)
        self.valueChanged.connect(self.skipper.set_num_frames)
