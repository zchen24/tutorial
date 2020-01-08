#!/usr/bin/env python3

"""
Shows how to show to use grid layout
"""

import sys
from qtpy.QtWidgets import *


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = QWidget()
    grid = QGridLayout()
    grid.addWidget(QPushButton("0/0"), 0, 0)
    grid.addWidget(QPushButton("0/1"), 0, 1)
    grid.addWidget(QPushButton("0/2"), 0, 2)
    grid.addWidget(QPushButton("1/0"), 1, 0)

    # span 2 columns
    row = 1
    col = 1
    rowSpan = 1
    colSpan = 2
    grid.addWidget(QPushButton("1/1+2"), row, col, rowSpan, colSpan)

    w.setLayout(grid)
    w.show()
    sys.exit(app.exec_())
