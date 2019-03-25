#!/usr/bin/env python

"""
Some QPushButton features
"""

import sys
from qtpy.QtWidgets import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()

        vbox = QVBoxLayout()
        pb1 = QPushButton('Option 1')
        pb1.setCheckable(True)
        pb1.setChecked(True)   # select rb1

        pb2 = QPushButton('Option 2')
        pb2.setCheckable(True)
        pb2.setStyleSheet("QPushButton:checked { background-color: green; }")

        vbox.addWidget(pb1)
        vbox.addWidget(pb2)
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyWidget()
    mw.show()
    sys.exit(app.exec_())
