#!/usr/bin/env python3

"""
Minimum QMediaPlayer example
Author: Zihan Chen
Date  : 2023-01-29
"""

import sys
import time
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mp = QMediaPlayer()
    mp.setMedia(QMediaContent(QUrl.fromLocalFile("/Users/tom/dev/tutorial/qt/assets/example.wav")))
    print("setMedia done")
    mp.play()
    time.sleep(5)
    sys.exit(app.exec_())
