#!/usr/bin/env python3

"""
Shows how to use a Qt QPrintDialog
"""

import sys
import time
from qtpy.QtWidgets import *
from qtpy.QtCore import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    num_files = 100
    progress = QProgressDialog('Copying Files', 'Abort', 0, num_files, parent=None)
    progress.setWindowModality(Qt.WindowModal)
    progress.show()

    for i in range(num_files):
        progress.setValue(i)
        if progress.wasCanceled():
            print('Copying files cancelled')
            break
        print('Copying file {:03}'.format(i))
        time.sleep(0.05)
    progress.setValue(num_files)
