#!/usr/bin/env python

"""
Shows how to inherit a QWidget
"""

import sys
from qtpy.QtWidgets import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()
        vbox = QVBoxLayout()
        vbox.addWidget(QPushButton('Button 1'))
        vbox.addWidget(QPushButton('Button 2'))
        vbox.addWidget(QPushButton('Button 3'))
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyWidget()
    mw.show()
    sys.exit(app.exec_())
