#!/usr/bin/env python3

"""
Shows how to show to use validator in line edit
e.g. A line editor that only accepts integer value
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWidget()
    edit_int = QLineEdit()
    int_min = 0
    int_max = 100
    edit_int.setValidator(QIntValidator(int_max, int_max))
    edit_double = QLineEdit()
    # min, max, decimal (int)
    edit_double.setValidator(QDoubleValidator(0, 50, int(2)))

    grid = QGridLayout()
    grid.addWidget(QLabel('Int Edit'), 0, 0)
    grid.addWidget(edit_int, 0, 1)
    grid.addWidget(QLabel('Double Edit'), 1, 0)
    grid.addWidget(edit_double, 1, 1)
    w.setLayout(grid)
    w.show()
    sys.exit(app.exec_())
