#!/usr/bin/env python

"""
Qt Groupbox Example

Author: Zihan Chen
Date: 2021-11-28
"""

import sys
from PyQt5.QtWidgets import *


if __name__ == '__main__':
    app = QApplication(sys.argv)

    qgb = QGroupBox("GroupBox")
    vbox = QVBoxLayout()
    vbox.addWidget(QPushButton("hello"))
    vbox.addWidget(QPushButton("world"))
    qgb.setLayout(vbox)
    qgb.setMinimumWidth(200)
    qgb.show()
    sys.exit(app.exec_())
