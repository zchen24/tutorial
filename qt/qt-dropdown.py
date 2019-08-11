#!/usr/bin/env python3

"""
Shows how to create a dropdown menu
"""

import sys
from qtpy.QtWidgets import *


class MyWidget(QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()

        combo = QComboBox()
        combo.addItem("Monday")
        combo.addItem("Tuesday")
        combo.addItem("Wednesday")
        combo.addItem("Thursday")
        combo.addItem("Friday")

        vbox = QVBoxLayout()
        vbox.addWidget(combo)
        self.setLayout(vbox)
        self.setWindowTitle("Dropdown Menu Demo")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyWidget()
    mw.show()
    sys.exit(app.exec_())
