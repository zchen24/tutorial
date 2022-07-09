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


def show_msg_box_question():
    print("Show MessageBox")
    QMessageBox.question(None, "Question", "This is a question", QMessageBox.Ok)

def show_msg_box_information():
    QMessageBox.information(None, "Info", "This is an information", QMessageBox.Ok)


def show_msg_box_warn():
    QMessageBox.warning(None, "Warn", "This is a warning", QMessageBox.Ok)


def show_msg_box_critical():
    QMessageBox.critical(None, "Critical", "This is a critical", QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWidget()
    w.setWindowTitle("QMessageBox Example")
    btn = QPushButton("Show message")
    btn.setMinimumWidth(200)
    btn.clicked.connect(show_msg_box)

    btn_ques = QPushButton("Show Question")
    btn_ques.clicked.connect(show_msg_box_question)
    btn_info = QPushButton("Show Info")
    btn_info.clicked.connect(show_msg_box_information)
    btn_warn = QPushButton("Show Warn")
    btn_warn.clicked.connect(show_msg_box_warn)
    btn_crit = QPushButton("Show Critical")
    btn_crit.clicked.connect(show_msg_box_critical)

    vbox = QVBoxLayout()
    vbox.addWidget(btn)
    vbox.addWidget(btn_ques)
    vbox.addWidget(btn_info)
    vbox.addWidget(btn_warn)
    vbox.addWidget(btn_crit)
    w.setLayout(vbox)
    w.show()
    sys.exit(app.exec_())
