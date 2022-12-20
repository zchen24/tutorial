#!/usr/bin/env python

"""
Shows how to use qslider
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()
        self.setWindowTitle('QSlider Demo')

        self.l = QLabel()
        # self.s = QSlider(Qt.Horizontal)
        self.s = QSlider()
        self.s.setOrientation(Qt.Horizontal)
        self.s.setMinimum(0)
        self.s.setMaximum(100)
        # self.s.setRange(0, 100)        
        self.s.setValue(50)
        self.s.valueChanged.connect(self.callback_slider)
        self.callback_slider()

        vbox = QVBoxLayout()
        vbox.addWidget(self.l)
        vbox.addWidget(self.s)
        self.setLayout(vbox)

    def callback_slider(self):
        self.l.setText('{}'.format(self.s.value()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())
