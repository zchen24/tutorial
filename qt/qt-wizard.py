#!/usr/bin/env python3

"""
QWizard example
Reference: https://www.youtube.com/watch?v=glaLe32OK8s

Date: 2022-06-05
"""

import sys
from PyQt5.QtWidgets import *


class Page1(QWizardPage):
    def __init__(self, parent):
        super(Page1, self).__init__(parent=parent)
        self.setTitle("Page 1")
        self.setSubTitle("This is page 1: please follow")

        tbox = QVBoxLayout()
        hbox = QHBoxLayout()
        lbl_name = QLabel("&Name")
        self.edit_name = QLineEdit()
        lbl_name.setBuddy(self.edit_name)
        hbox.addWidget(lbl_name)
        hbox.addWidget(self.edit_name)
        tbox.addLayout(hbox)

        hbox = QHBoxLayout()
        lbl_address = QLabel("&Address")
        edit_address = QLineEdit()
        lbl_address.setBuddy(edit_address)
        hbox.addWidget(lbl_address)
        hbox.addWidget(edit_address)
        tbox.addLayout(hbox)
        self.setLayout(tbox)

        # fields
        self.registerField("Name*", self.edit_name)
        self.registerField("Address", edit_address)


class Page2(QWizardPage):
    def __init__(self, parent):
        super(Page2, self).__init__(parent=parent)
        self.setTitle("Page 2")
        self.setSubTitle("This is page 2: please follow")

        tbox = QVBoxLayout()
        self.lbl_name = QLabel("Name")
        tbox.addWidget(self.lbl_name)
        self.setLayout(tbox)

    def initializePage(self) -> None:
        """
        This is called when Page 2 is about to be shown
        """
        name = self.field("Name")
        self.lbl_name.setText("Name: {}".format(name))
        return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWizard()
    w.setWindowTitle("QMessageBox Example")
    w.addPage(Page1(w))
    w.addPage(Page2(w))

    # custom button text: uncomment to use
    # w.setButtonText(QWizard.BackButton, "上一步")
    # w.setButtonText(QWizard.NextButton, "下一步")
    # w.setButtonText(QWizard.CancelButton, "取消")
    # w.setButtonText(QWizard.FinishButton, "结束")
    print(w.pageIds())
    w.show()
    sys.exit(app.exec_())
