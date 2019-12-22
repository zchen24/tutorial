#!/usr/bin/env python3

"""
Shows how to show widget in full screen mode
"""

import sys
from qtpy.QtWidgets import *
from qtpy.QtGui import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    img = QImage('../opencv/imgs/lena.jpg')
    w = QLabel()
    w.setPixmap(QPixmap(img))

    # NOTE: show in fullscreen
    w.showFullScreen()
    sys.exit(app.exec_())
