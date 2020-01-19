#!/usr/bin/env python

"""
Some QSplitter features
"""

import sys
from qtpy.QtWidgets import *
from qtpy.QtCore import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()

        vbox = QVBoxLayout()
        pb1 = QPushButton('Option 1')
        pb2 = QPushButton('Option 2')
        pb3 = QPushButton('Option 3')
        pb4 = QPushButton('Option 4')

        # split horizontally
        h_splitter = QSplitter()
        h_splitter.addWidget(pb1)
        h_splitter.addWidget(pb2)

        # split vertically
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(pb3)
        v_splitter.addWidget(pb4)

        vbox.addWidget(h_splitter)
        vbox.addWidget(v_splitter)
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyWidget()
    mw.show()
    sys.exit(app.exec_())
