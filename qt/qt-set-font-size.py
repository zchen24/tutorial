#!/usr/bin/env python

"""
Shows how to set font size
Date: 2022-06-09
"""

import sys
from qtpy.QtWidgets import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = app.font()
    # change font to 20
    font.setPointSize(20)
    app.setFont(font)
    w = QLabel("Hello")
    w.show()
    sys.exit(app.exec_())
