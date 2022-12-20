#!/usr/bin/env python

"""
Shows how to use radio button group (exclusive)
"""

import sys
from PyQt5.QtWidgets import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()

        vbox = QVBoxLayout()
        gbox = QGroupBox('Exclusive Radio Buttons')

        rb1 = QRadioButton('Option 1')
        rb2 = QRadioButton('Option 2')
        rb3 = QRadioButton('Option 3')
        vbox.addWidget(rb1)
        vbox.addWidget(rb2)
        vbox.addWidget(rb3)
        gbox.setLayout(vbox)
        rb1.setChecked(True)   # select rb1

        vbox = QVBoxLayout()
        vbox.addWidget(gbox)
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyWidget()
    mw.show()
    sys.exit(app.exec_())
