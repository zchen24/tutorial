#!/usr/bin/env python

import sys
from PyQt5 import QtWidgets, QtCore, uic

class QtPythonGui(QtWidgets.QMainWindow):
    def __init__(self):
        super(QtPythonGui, self).__init__()
        uic.loadUi('main.ui', self)
        self.setWindowTitle(self.tr('Example UI GUI'))
        self._timer = QtCore.QTimer()
        self._timer.start(50)
        self._timer.timeout.connect(self._update)
        
    def keyPressEvents(self, event):
        print("key pressed")

    def mousePressEvent(self, mouse_event):
        if mouse_event.button() == QtCore.Qt.LeftButton:
            print("left mouse button pressed")
        
    def _update(self):
        # print("... updating ...")
        pass            
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = QtPythonGui()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()



