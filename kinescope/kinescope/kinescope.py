import argparse
import sys

from PyQt5.QtWidgets import QApplication

from project_window import ProjectWindow
from main_window import MainWindow
from utils import centerWindow


class Windows(object):

    def __init__(self, app, is_simulation):

        self.app = app

        self.project_window = ProjectWindow()
        self.main_window = MainWindow('', is_simulation=is_simulation, include_nn=True)

        self.app.aboutToQuit.connect(self.main_window.cleanup)

        self.project_window.project_dir_callback = self.start_main
        self.project_window.show()
        centerWindow(self.project_window)

    def start_main(self, project_dir):
        self.main_window.set_project_dir(project_dir)
        self.project_window.close()
        self.main_window.show()
        self.main_window.initializeGeometry()
        self.main_window.initialize_trigger()
        self.main_window.start_grabbing()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='kinescope')
    parser.add_argument('--sim-camera', default=False, type=bool, help='camera is simulated')
    args = parser.parse_args()

    app = QApplication(sys.argv[0:1])
    windows = Windows(app, args.sim_camera)
    app.exec()
