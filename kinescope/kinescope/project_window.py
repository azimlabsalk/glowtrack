from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QAction, QDesktopWidget, QDockWidget,
    QFileDialog,  QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMenu, QPushButton, QSizePolicy, QScrollArea,
    QSpinBox, QTreeView, QWidget, QVBoxLayout)

class ProjectWindow(QMainWindow):

    def __init__(self):
        super(ProjectWindow, self).__init__()

        self.setWindowTitle("kinescope")
        self.setWindowIcon(QIcon("images/telescope.svg"))

        self.createWidgets()

    def createWidgets(self):
        self.new_button = QPushButton(QIcon("images/plus-square.svg"), "New Project")
        self.open_button = QPushButton(QIcon("images/edit.svg"), "Open Project...")

        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.new_button)
        self.layout.addWidget(self.open_button)

        self.new_button.clicked.connect(self.new)
        self.open_button.clicked.connect(self.open)

        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def new(self):
        project_dir = QFileDialog.getExistingDirectory(None, "New Project", "~/",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)

        self.project_dir_callback(project_dir)

    def open(self):
        project_dir = QFileDialog.getExistingDirectory(None, "Open Project", "~/",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)

        if self.project_valid(project_dir):
            self.project_dir_callback(project_dir)
        else:
            print('Invalid project directory')

    def project_valid(self, project_dir):
        return True

