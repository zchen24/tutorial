#!/usr/bin/env python

"""
Shows how to display a image in Qt Widget
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    img = QImage('../opencv/imgs/lena.jpg')
    w = QLabel()
    w.setPixmap(QPixmap(img))
    w.show()
    sys.exit(app.exec_())
