#!/usr/bin/env python

"""
Some QPushButton features
"""

import sys
from qtpy.QtWidgets import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()

        vbox = QVBoxLayout()
        pb1 = QPushButton('&Option 1')
        pb1.setCheckable(True)
        pb1.clicked.connect(self.slot_pb1_clicked)

        pb2 = QPushButton('Option 2')
        pb2.setShortcut("Ctrl+N")
        pb2.clicked.connect(self.slot_pb2_clicked)

        vbox.addWidget(pb1)
        vbox.addWidget(pb2)
        self.setLayout(vbox)

    def slot_pb1_clicked(self):
        print("pb1 clicked")

    def slot_pb2_clicked(self):
        print("pb2 clicked")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyWidget()
    mw.show()
    sys.exit(app.exec_())
