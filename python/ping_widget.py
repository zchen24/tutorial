#!/usr/bin/env python3

"""
Ping QtWidget using fping pkg

sudo apt install python3-pyqt5 fping
"""


import platform
import subprocess
import sys

import logging
from PyQt5.QtWidgets import *


def ping_host(host):
    parameter = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['fping', parameter, '1', '-t', '200', host]
    try:
        response = subprocess.call(command)
        if response == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        logging.error("fping not found. Please install sudo apt install fping")
        return False


class PingPushButton(QPushButton):
    def __init__(self, name='A-Host', host='192.168.1.2', parent=None):
        super(PingPushButton, self).__init__(name + ' ' + host, parent)
        self.name = name
        self.host = host
        self.clicked.connect(self.ping_target)
        self.setStyleSheet(
            "QPushButton:hover { background-color: lightgreen; } "
            "QPushButton:checked { background-color: green; } "
            "QPushButton:disabled { color: black; }")

    def ping_target(self):
        if ping_host(self.host):
            logging.info("ping {} success".format(self.host))
            self.setStyleSheet(self.styleSheet() + ' QPushButton {background-color: green}')
        else:
            logging.warning("ping {} fail".format(self.host))
            self.setStyleSheet(self.styleSheet() + ' QPushButton {background-color: red}')
            self.styleSheet()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWidget()
    vbox = QVBoxLayout()
    vbox.addWidget(PingPushButton('A-Host', '192.168.1.1'))
    vbox.addWidget(PingPushButton('B-Host', '192.168.1.2'))
    vbox.addWidget(PingPushButton('C-Host', '192.168.1.3'))
    vbox.addWidget(PingPushButton('D-Host', '192.168.1.4'))
    w.setLayout(vbox)
    w.show()
    app.exec_()