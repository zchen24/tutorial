#!/usr/bin/env python

"""
Shows how to display a image in Qt Widget
"""

import sys
from qtpy.QtWidgets import *
from qtpy.QtGui import *
from qtpy.QtCore import *
import cv2


class QWebcam(QLabel):
    def __init__(self):
        super(QWebcam, self).__init__()
        self.cam = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(50)  # 20 hz
        self.setGeometry(0, 0, 800, 600)

    def __del__(self):
        self.cam.release()

    def update_video(self):
        _, frame = self.cam.read()
        h, w, c = frame.shape
        qimg = QImage(frame.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self.setPixmap(QPixmap(qimg))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWebcam()
    w.show()
    sys.exit(app.exec_())
