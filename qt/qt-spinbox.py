#!/usr/bin/env python

"""
Shows how to use Hex Spinbox
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class QHexSpinBox(QSpinBox):
    def __init__(self):
        super(QHexSpinBox, self).__init__()
        self.setRange(0x000, 0x3FF)
        self.validator = QRegExpValidator(QRegExp("[0-9A-Fa-f]{1,8}"), self)

    def validate(self, txt: str, pos: int):
        return self.validator.validate(txt, pos)

    def valueFromText(self, text: str):
        return int(text, 16)

    def textFromValue(self, v: int):
        return hex(v).upper()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    s = QHexSpinBox()
    s.setRange(0x000, 0xFFF)
    s.setValue(0x7FF)
    s.setPrefix('  ')
    s.setGeometry(0, 0, 250, 70)
    s.setWindowTitle('Hex Spinbox')
    s.show()
    sys.exit(app.exec_())
