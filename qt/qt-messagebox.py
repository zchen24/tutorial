#!/usr/bin/env python3

"""
Shows how to use QMessageBox

Date: 2022-06-05
"""

import sys
from PyQt5.QtWidgets import *

def show_msg_box():
    print("Show MessageBox")
    mbox = QMessageBox()
    mbox.setIcon(QMessageBox.Information)
    mbox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    mbox.setWindowTitle("Info Box")
    mbox.setText("Please click the two option buttons")
    ret = mbox.exec()
    if ret == QMessageBox.Ok:
        print("OK Clicked")
    elif ret == QMessageBox.Cancel:
        print("Cancel Clicked")
    else:
        print("Invalid")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWidget()
    w.setWindowTitle("QMessageBox Example")
    btn = QPushButton("Show message")
    btn.setMinimumWidth(200)
    btn.clicked.connect(show_msg_box)
    vbox = QVBoxLayout()
    vbox.addWidget(btn)
    w.setLayout(vbox)
    w.show()
    sys.exit(app.exec_())
