#!/usr/bin/env python3

"""
Some QPushButton features
"""

import sys
from qtpy.QtWidgets import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()

        tbox = QVBoxLayout()
        hbox = QHBoxLayout()
        lbl_phone = QLabel("&Phone")
        edit_phone = QLineEdit()
        lbl_phone.setBuddy(edit_phone)
        hbox.addWidget(lbl_phone)
        hbox.addWidget(edit_phone)
        tbox.addLayout(hbox)

        hbox = QHBoxLayout()
        lbl_address = QLabel("&Address")
        edit_address = QLineEdit()
        lbl_address.setBuddy(edit_address)
        hbox.addWidget(lbl_address)
        hbox.addWidget(edit_address)
        tbox.addLayout(hbox)

        text_help = QTextEdit()
        text_help.append("Alt+P to switch focus to Phone")
        text_help.append("Alt+A to switch focus to Address")
        tbox.addWidget(text_help)
        self.setLayout(tbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyWidget()
    mw.show()
    sys.exit(app.exec_())
