#!/usr/bin/env python

import sys
from qtpy import QtWidgets



def window():
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    b = QtWidgets.QLabel(w)
    b.setText("Hello World!")
    w.setGeometry(100,100,200,50)
    b.move(50,20)
    w.setWindowTitle("hello")
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    window()
