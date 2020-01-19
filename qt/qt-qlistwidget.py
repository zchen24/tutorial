#!/usr/bin/env python

"""
Some QListWidget features
Reference:
https://medium.com/xster-tech/pyqt-drag-images-into-list-widget-for-thumbnail-list-e4a12f906bd8
"""

import sys
from qtpy.QtWidgets import *
from qtpy.QtCore import *
from qtpy.QtGui import *
from PIL import Image, ImageQt


class MyListWidget(QListWidget):
    def __init__(self):
        super(MyListWidget, self).__init__()
        self.setIconSize(QSize(100, 100))
        self.setAcceptDrops(True)

        img = Image.open('./assets/icon.png')
        img.thumbnail((72, 72), Image.ANTIALIAS)

        icon = QIcon(QPixmap.fromImage(ImageQt.ImageQt(img)))
        self.item1 = QListWidgetItem(icon, 'Image 1', self)
        self.item2 = QListWidgetItem(icon, 'Image 2', self)
        self.setWindowTitle('Data Browser')
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyListWidget()
    mw.show()
    sys.exit(app.exec_())
