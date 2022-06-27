from PyQt5.QtWidgets import QDesktopWidget

def centerWindow(window):
    qr = window.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    window.move(qr.topLeft())

