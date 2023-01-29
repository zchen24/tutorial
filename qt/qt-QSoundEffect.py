#!/usr/bin/env python3

"""
Minimum QSoundEffect example
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
    se = QSoundEffect()
    se.setSource(QUrl.fromLocalFile(os.path.abspath("./assets/example.wav")))
    print("setMedia done")
    se.play()
    se.setLoopCount(2)
    se.setVolume(0.5)
    time.sleep(5)
    sys.exit(app.exec_())
