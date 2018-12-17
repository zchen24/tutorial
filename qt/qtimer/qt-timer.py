#!/usr/bin/env python

"""
Example shows how to use a QTimer

"""

import sys
from qtpy import QtWidgets
from qtpy import QtCore


def timer_update():
    print('timer update called')


def window():
    app = QtWidgets.QApplication(sys.argv)
    timer = QtCore.QTimer()
    timer.timeout.connect(timer_update)
    timer.start(500)   # in ms

    btn = QtWidgets.QPushButton("QTimer Example")
    btn.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    window()
