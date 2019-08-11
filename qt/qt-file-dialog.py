#!/usr/bin/env python3

"""
Shows how to use a Qt FileDialog
"""

import sys
from qtpy.QtWidgets import *


if __name__ == '__main__':
    app = QApplication(sys.argv)

    options = QFileDialog.Options()
    caption = "Title: Getting File Name"
    directory = "./"
    filters = "All Files (*);;Python Scripts (*.py)"
    initial_filter = "Python Scripts (*.py)"
    filename, _ = QFileDialog.getOpenFileName(None,
                                              caption,
                                              directory,
                                              filters,
                                              initial_filter,
                                              options)
    sys.exit(app.exec_())
