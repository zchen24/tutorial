#!/usr/bin/env python

"""
Shows how to display a image in Qt Widget
"""

import sys
from qtpy.QtWidgets import *
from qtpy.QtGui import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    img = QImage('../opencv/imgs/lena.jpg')
    w = QLabel()
    w.setPixmap(QPixmap(img))
    w.show()
    appIcon = QIcon('../qt/assets/icon.png')
    app.setWindowIcon(appIcon)
    sys.exit(app.exec_())
