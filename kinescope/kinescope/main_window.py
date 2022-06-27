from collections import namedtuple
import sys
import threading
import time

import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt, QObject
from PyQt5.QtGui import QIcon, QImage, QPalette, QPixmap, QStandardItemModel
from PyQt5.QtWidgets import (QAction, QDesktopWidget, QDockWidget,
    QFileDialog,  QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMenu, QPushButton, QSizePolicy, QScrollArea,
    QSpinBox, QTreeView, QWidget, QVBoxLayout)

from q_arduino_camera_trigger import QArduinoCameraTrigger
from q_board_camera_trigger import QBoardCameraTrigger
from detect_dye_stage import DetectDyeStage
from data_explorer_widget import DataExplorerWidget
from controls_widget import ControlsWidget
from clip_player import ClipPlayer
from image_widget import ImageWidget
#from cyclegan_stage import CycleGanStage
from realtime_feedback_widget import RealtimeFeedbackWidget
from realtime_image_pipeline import Flipper
from realtime_pipeline import Buffer, Indexer, Pairer, Skipper, Stage, StatelessStage
from realtime_pipeline_qt import DataStorer, FrameToNumpyImage, NumpyImageToQImageSignal
import settings
from simulated_camera import SimulatedCamera
from simulated_camera_trigger import SimulatedCameraTrigger
from static_camera import StaticCamera
from utils import centerWindow
from video_data_store import VideoDataStore


class MainWindow(QMainWindow):

    def __init__(self, project_dir, width=1200, height=800, is_simulation=False, trigger_type='none', include_nn=True):
        super(MainWindow, self).__init__()

        self.setWindowTitle("kinescope")
        self.setWindowIcon(QIcon("images/telescope.svg"))

        self.is_simulation = is_simulation
        self.include_nn = include_nn
        self.project_dir = project_dir
        self.is_recording = False
        self.is_triggering = False
        self.clip_index = None
        self.trigger_type = trigger_type

        self.createDataStore(self.project_dir)
        self.createDataPipeline()
        self.createDockWindows()
        self.createActions()
        self.createMenus()
        self.createToolBars()

        self.init_width = width
        self.init_height = height

    def set_project_dir(self, project_dir):
        self.project_dir = project_dir
        self.dataStore.set_project_dir(self.project_dir)
        self.setWindowTitle("kinescope - " + project_dir)

    def initializeGeometry(self):
        self.setGeometry(0, 0, self.init_width, self.init_height)
        centerWindow(self)

    def createDataStore(self, project_dir):
        self.dataStore = VideoDataStore(project_dir)

    def createActions(self):
        self.toggleRecordingAction = QAction("&Start/stop recording", self, shortcut=Qt.Key_Space,
            triggered=self.toggleRecording)
        # self.toggleRecordingAction.setShortcutVisibleInContextMenu(True)

        #self.openAction = QAction("&Open...", self, shortcut="Ctrl+O",
        #        triggered=self.open)

        self.quitAction = QAction("&Quit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.updateActions()

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        #self.fileMenu.addAction(self.openAction)
        self.fileMenu.addAction(self.quitAction)
        # self.fileMenu.addAction(self.printAct)
        # self.fileMenu.addSeparator()

        self.cameraMenu = QMenu("&Camera", self)
        self.cameraMenu.addAction(self.toggleRecordingAction)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.cameraMenu)

    def toggleTriggering(self):
        if not self.is_triggering:
            self.start_triggering()
        else:
            self.stop_triggering()
        self.is_triggering = not self.is_triggering
        self.updateActions()

    def start_triggering(self):
        self.trigger.start_triggering()

    def stop_triggering(self):
        self.trigger.stop_triggering()
        self.pairer.clear()
        self.buffer.clear()
        self.skipper.clear()

    def toggleRecording(self):
        if not self.is_recording:
            self.start_triggering()
            self.startRecording()
        else:
            self.stopRecording()
        self.is_recording = not self.is_recording
        self.updateActions()

    def startRecording(self):
        self.clip_index = self.dataStore.createClip()
        self.data_storer.startClip(self.clip_index)
        self.buffer.flush()
        self.buffer.buffering = False

    def stopRecording(self):
        self.data_storer.stopClip()
        self.dataStore.clipFinished(self.clip_index)
        self.clip_index = None
        self.buffer.buffering = True

    def updateActions(self):
        if self.is_recording:
            recording_icon = QIcon('images/camera-red.png')
        else:
            recording_icon = QIcon('images/camera-blue.png')
        self.toggleRecordingAction.setIcon(recording_icon)

    def open(self):
        pass

    def createToolBars(self):
        self.cameraToolBar = self.addToolBar("Camera")
        self.cameraToolBar.addAction(self.toggleRecordingAction)

    def initialize_trigger(self):
        self.trigger.initialize()

    def createDockWindows(self):

        self.setDockNestingEnabled(True)
        # self.centralPlaceholder = QLabel()
        # self.setCentralWidget(self.centralPlaceholder)
        # self.centralPlaceholder.setMaximumSize(0,0)
        # self.centralPlaceholder.setSizePolicy(QSizePolicy.Minimum,
        #     QSizePolicy.Minimum)

        # controls
        self.controls_widget = ControlsWidget("Controls", self,
            include_nn=self.include_nn)
        self.controls_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.controls_widget.camera_controls.buffer_size_widget.set_buffer(self.buffer)
        self.controls_widget.camera_controls.skip_frames_widget.set_skipper(self.skipper)
        self.controls_widget.camera_controls.flip_lr_checkbox.toggled.connect(self.flipper_stage.set_flip_lr)
        self.controls_widget.camera_controls.flip_ud_checkbox.toggled.connect(self.flipper_stage.set_flip_ud)

        self.controls_widget.data_controls.clip_name.setText('clip_{}')
        self.controls_widget.data_controls.clip_name.textChanged.connect(
            self.dataStore.setClipNameTemplate)

        if self.include_nn:
            self.controls_widget.nn_controls.loadNet.connect(self.neural_network_stage.load)
            self.controls_widget.nn_controls.conf_threshold_widget.set_thresholder(
                self.neural_network_stage)

        # configure trigger controls

        def set_duration(ms):
            self.trigger.set_duration(ms)

        cycle_duration = self.controls_widget.trigger_controls.cycle_duration
        cycle_duration.setValue(10.0)
        cycle_duration.valueChanged.connect(set_duration)

        init_values = [0.0, 2.5, 2.5, 4.5, 5.0, 10.0, 7.5, 9.5]

        for i in range(8):
            timebox = self.controls_widget.trigger_controls.timeboxes[i]
            timebox.setValue(init_values[i])
            func = set_trigger_time(self.trigger, i, self)
            timebox.valueChanged.connect(func)

        # self.controls_widget.setAllowedAreas(Qt.LeftDockWidgetArea)
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.controls_widget)

        # camera views
        self.image_widget_uv = ImageWidget("UV Camera", self)
        self.image_widget_uv.setAllowedAreas(Qt.RightDockWidgetArea)
        self.image_to_qimage_uv.qimage.connect(self.image_widget_uv.setImage)

        self.image_widget_visible = ImageWidget("Visible Camera", self)
        self.image_widget_visible.setAllowedAreas(Qt.RightDockWidgetArea)
        self.image_to_qimage_visible.qimage.connect(self.image_widget_visible.setImage)

        if self.include_nn:
            self.image_widget_nn = ImageWidget("Neural Net (Image)", self)
            self.image_widget_nn.setAllowedAreas(Qt.RightDockWidgetArea)
            self.image_to_qimage_nn.qimage.connect(self.image_widget_nn.setImage)

        self.image_widget_mask = ImageWidget("UV Mask Playback", self)
        self.image_widget_mask.setAllowedAreas(Qt.RightDockWidgetArea)
        self.image_to_qimage_mask.qimage.connect(self.image_widget_mask.setImage)

        self.addDockWidget(Qt.RightDockWidgetArea, self.image_widget_uv)
        self.tabifyDockWidget(self.image_widget_uv, self.image_widget_visible)
        if self.include_nn:
            self.tabifyDockWidget(self.image_widget_uv, self.image_widget_nn)
        self.tabifyDockWidget(self.image_widget_uv, self.image_widget_mask)

        # data
        self.data_explorer_widget = DataExplorerWidget("Data", self)
        self.data_explorer_widget.setModel(self.dataStore.itemModel())
        self.data_explorer_widget.new_selection.connect(self.clip_player.handleNewSelection)
        self.data_explorer_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.data_explorer_widget.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.data_explorer_widget)

        self.setCentralWidget(self.controls_widget)
        self.controls_widget.resize(200, 500)

        self.data_explorer_widget.setFocus()

        # realtime feedback
        self.realtime_feedback_widget = RealtimeFeedbackWidget("Realtime Feedback", self)
        self.nn_xy_indexer.connect(self.realtime_feedback_widget)
        self.tabifyDockWidget(self.data_explorer_widget, self.realtime_feedback_widget)

    def createDataPipeline(self):

        if self.is_simulation:
            print('creating simulated camera')
            self.camera = SimulatedCamera(1440, 1080, 100, alternating=True)
            #self.camera = StaticCamera(('/home/djbutler/dev/'
            #                            'pytorch-CycleGAN-and-pix2pix/datasets'
            #                            '/Gen1_1-400/testA/Reach_180222_008001'
            #                            '.png'), 100, alternating=True)
            self.trigger = SimulatedCameraTrigger()
        else:
            print('creating basler camera')
            from basler_camera import BaslerCamera
            external_trigger = self.trigger_type != 'none'
            self.camera = BaslerCamera(trigger_mode=external_trigger)
            print(self.camera.mono)
            triggers = {
                'arduino': QArduinoCameraTrigger,
                'board': QBoardCameraTrigger,
                'none': SimulatedCameraTrigger
            }
            self.trigger = triggers[self.trigger_type]()

        self.flipper_stage = Flipper()
        self.pairer = Pairer()
        self.skipper = Skipper(settings.INIT_FRAMES_TO_SKIP)
        self.buffer = Buffer(settings.INIT_BUFFER_SIZE, True)
        self.data_storer = DataStorer(self.dataStore)

        self.clip_player = ClipPlayer(self.dataStore)

        self.indexer_visible = Indexer(settings.VISIBLE_INDEX)
        self.frame_to_image_visible = FrameToNumpyImage()
        self.image_to_qimage_visible = NumpyImageToQImageSignal()

        self.indexer_uv = Indexer(settings.UV_INDEX)
        self.frame_to_image_uv = FrameToNumpyImage()
        self.image_to_qimage_uv = NumpyImageToQImageSignal()

        self.nn_image_indexer = Indexer(0)
        self.nn_xy_indexer = Indexer(1)

        self.camera.connect(self.flipper_stage)
        self.flipper_stage.connect(self.pairer)
        self.pairer.connect(self.skipper)
        self.skipper.connect(self.buffer)
        self.buffer.connect(self.data_storer)

        # hook up uv image widget
        self.pairer.connect(self.indexer_uv)
        self.indexer_uv.connect(self.frame_to_image_uv)
        self.frame_to_image_uv.connect(self.image_to_qimage_uv)

        # hook up visible image widget
        self.pairer.connect(self.indexer_visible)
        self.indexer_visible.connect(self.frame_to_image_visible)
        self.frame_to_image_visible.connect(self.image_to_qimage_visible)

        # hook up the neural network stage
        if self.include_nn:
            from deepercut_stage import DeeperCutStage
            self.neural_network_stage = DeeperCutStage()
            # self.neural_network_stage = CycleGanStage()
            self.image_to_qimage_nn = NumpyImageToQImageSignal()
            self.frame_to_image_visible.connect(self.neural_network_stage)

            self.neural_network_stage.connect(self.nn_image_indexer)
            self.neural_network_stage.connect(self.nn_xy_indexer)

            self.nn_image_indexer.connect(self.image_to_qimage_nn)

        # hook up mask stage
        self.dye_detector = DetectDyeStage()
        self.clip_player.connect(self.dye_detector)
        self.image_to_qimage_mask = NumpyImageToQImageSignal()
        self.dye_detector.connect(self.image_to_qimage_mask)

    def start_grabbing(self):
        self.camera.start_grabbing()
        # self.start_triggering()
        # self.grab_timer = threading.Timer(5.0, self.start_triggering)
        # self.grab_timer.start()

    def stop_grabbing(self):
        self.camera.stop_grabbing()

    def closeEvent(self, event):
        if self.dataStore.is_busy():
            event.ignore()
        else:
            self.cleanup()
            event.accept() # let the window close

    def cleanup(self):
        self.stop_triggering()
        self.stop_grabbing()

def set_trigger_time(trigger, idx, main_window):
    def func(ms):
        main_window.stop_triggering()
        trigger.set_time(idx, ms)
        main_window.start_triggering()
    return func
