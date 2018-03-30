#!/usr/bin/env python

"""
Demonstrate how to integrate Matplotlib canvas into Qt

Zihan Chen
2018-03-29
"""


# get canvas etc.
from __future__ import print_function
import sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *


class MplCanvas(FigureCanvas):
    """
    QWidget and FigureCanvas
    """
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(1, 1, 1)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        x = np.arange(0.0, 10.0, 0.1)
        y = np.sin(x)
        self.axes.plot(x, y)
        self.axes.set_title('Example in Qt')
        self.axes.set_xlabel('time (s)')
        self.axes.set_ylabel('value')
        self.axes.grid(True)


class MplWidgetWithToolbar(QWidget):
    def __init__(self, parent=None):
        super(MplWidgetWithToolbar, self).__init__(parent)
        canvas = MplCanvas()
        toolbar = NavigationToolbar2QT(canvas, self)
        vbox = QVBoxLayout()
        vbox.addWidget(toolbar)
        vbox.addWidget(canvas)
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MplWidgetWithToolbar()
    mw.show()
    sys.exit(app.exec_())
