#!/usr/bin/env python

"""
Shows how to develop QMainWindow
- Menu (e.g. help)
- Actions
- Toolbar
"""


import sys
from qtpy.QtWidgets import *
from qtpy.QtGui import *


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        w = QLabel()
        img = QImage('../opencv/imgs/lena.jpg')
        w.setPixmap(QPixmap(img))

        file_menu = self.menuBar().addMenu('File')

        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit Application')
        exitAction.triggered.connect(self.close)
        file_menu.addAction(exitAction)

        tool_bar = self.addToolBar("File")
        tool_bar.addAction(exitAction)

        self.setCentralWidget(w)
        self.setWindowTitle("My Main Window")

    def __del__(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_win = MyWindow()
    my_win.show()
    sys.exit(app.exec_())