#!/usr/bin/env python

import sys

# Workaround for QOpenGLShaderProgram issue
# See: drammock's comment
# https://github.com/spyder-ide/spyder/issues/3226 
from OpenGL import GL

from PyQt5.QtCore import QObject,QUrl
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQml import qmlRegisterType
from PyQt5.QtQuick import QQuickView


class Person(QObject):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        
def main():
    print("start")
    app = QApplication(sys.argv)
    qmlRegisterType(Person, 'People', 1, 0, 'Person')
    v = QQuickView(QUrl("main.qml"))
    v.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

