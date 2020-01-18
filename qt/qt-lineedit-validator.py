#!/usr/bin/env python3

"""
Shows how to show to use validator in line edit
e.g. A line editor that only accepts integer value
"""

import sys
from qtpy.QtWidgets import *
from qtpy.QtGui import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWidget()
    edit_int = QLineEdit()
    int_min = 0
    int_max = 100
    edit_int.setValidator(QIntValidator(int_max, int_max))
    hbox = QHBoxLayout()
    hbox.addWidget(QLabel('Int Validator'))
    hbox.addWidget(edit_int)
    w.setLayout(hbox)
    w.show()
    sys.exit(app.exec_())
