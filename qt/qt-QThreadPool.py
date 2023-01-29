#!/usr/bin/env python3

"""
Some QThreadPool
https://doc-snapshots.qt.io/qt5-5.15/qthreadpool.html
"""

import sys
import time
from PyQt5.QtCore import *

class HelloWorldTask(QRunnable):
    def run(self):
        for i in range(10):
            print("Thread: {}, hello world - {}".format(QThread.currentThreadId(), i))
            time.sleep(0.5)


if __name__ == '__main__':
    app = QCoreApplication(sys.argv)
    task1 = HelloWorldTask()
    QThreadPool.globalInstance().start(task1)
    print("Pool activeThreadCount: {}".format(QThreadPool.globalInstance().activeThreadCount()))

    time.sleep(2.0)
    task2 = HelloWorldTask()
    QThreadPool.globalInstance().start(task2)
    print("Pool activeThreadCount: {}".format(QThreadPool.globalInstance().activeThreadCount()))
    app.exec()
