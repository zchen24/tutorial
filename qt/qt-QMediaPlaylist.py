#!/usr/bin/env python3

"""
Minimum QMediaPlayer example
Author: Zihan Chen
Date  : 2023-01-29
"""

import sys
import time
import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mp = QMediaPlayer()
    pl = QMediaPlaylist()
    mp.setPlaylist(pl)
    pl.addMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath("./assets/example.wav"))))
    pl.addMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath("./assets/example.mp3"))))

    pl.setPlaybackMode(QMediaPlaylist.CurrentItemOnce)
    pl.setCurrentIndex(0)
    mp.play()
    time.sleep(5)
    sys.exit(app.exec_())
