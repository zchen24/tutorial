#!/usr/bin/env python

"""
Shows how to use button group (exclusive)
"""

import sys
from qtpy.QtWidgets import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()

        vbox = QVBoxLayout()
        rb1 = QPushButton('Option 1')
        rb2 = QPushButton('Option 2')
        rb3 = QPushButton('Option 3')
        rb1.setCheckable(True)
        rb2.setCheckable(True)
        rb3.setCheckable(True)

        # IMPORTANT:
        # parent = self, don't miss this
        # otherwise, the btn_group will be deleted
        btn_group = QButtonGroup(self)
        btn_group.addButton(rb1)
        btn_group.addButton(rb2)
        btn_group.addButton(rb3)

        vbox.addWidget(rb1)
        vbox.addWidget(rb2)
        vbox.addWidget(rb3)
        rb1.setChecked(True)   # select rb1
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyWidget()
    mw.show()
    sys.exit(app.exec_())
