#!/usr/bin/env python

import sys

# Workaround for QOpenGLShaderProgram issue
# See: drammock's comment
# https://github.com/spyder-ide/spyder/issues/3226 
from OpenGL import GL

from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication


def main():
    print("start")
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.load(QUrl("main.qml"))

    engine.rootObjects()[0].show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

