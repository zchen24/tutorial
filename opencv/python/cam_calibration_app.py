#!/usr/bin/env python

"""
Camera Calibration Application

Author: Zihan Chen
Date: 2019-08-11
License: LGPL
"""


import sys
import glob
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from qtpy.QtWidgets import *
from qtpy.QtGui import *
from qtpy.QtCore import *


# Todo list:
#  - add GUI for mono calibration
#  - minic the Matlab's GUI

# Future release
#  - generate PDF checker file from Python class


# Usage:
#   s: save an image
#   c: calibrate
#   "Save Button"
#   "Calibrate button"
#   the result is saved in a ROS compatible format (yaml)


class CheckerBoardInfo(object):
    def __init__(self, cols, rows, square):
        self.cols = cols
        self.rows = rows
        self.dim = square

    def __str__(self):
        return "Checkerboard: \n width: {}\n height: {}\n square: {}".format(
            self.cols, self.rows, self.dim)

    def _check_value(self, value):
        if value <= 0:
            raise ValueError("Value must be a positive integer")
        else:
            return value

    @property
    def size(self):
        return (self.cols, self.rows)

    @property
    def cols(self):
        return self._cols

    @cols.setter
    def cols(self, value):
        self._cols = self._check_value(value)

    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = self._check_value(value)

    @property
    def dim(self):
        return self._square

    @dim.setter
    def dim(self, value):
        self._square = self._check_value(value)

    def make_object(self):
        # type must be float32 as calibrator expects Point3f type
        opts = np.zeros((self.rows*self.cols, 1, 3), dtype=np.float32)
        for jj, opt in enumerate(opts):
            opt[0, 0] = int(jj / self.cols)
            opt[0, 1] = int(jj % self.cols)
            opt[0, 2] = 0
        opts *= self.dim
        return opts

    def generate_pdf(self):
        """Generate a PDF file"""
        print("To Implement: generating a PDF file for printing")


class CameraInfoMono(object):
    def __init__(self, file_name = None):
        self.image_width = None
        self.image_height = None
        self.camera_name = None
        self.K = np.eye(3)
        self.D = np.zeros((1, 5))
        self.distortion_model = None
        self.R = np.eye(3)
        self.P = np.zeros((3,4))
        self.cfg = None  # yaml cdf dictionary
        if file_name is not None:
            self.read_yaml(file_name)

    @staticmethod
    def _load_matrix(cfg:dict, name):
        rows = cfg[name]['rows']
        cols = cfg[name]['cols']
        data = cfg[name]['data']
        return np.array(data).reshape((rows, cols))

    @staticmethod
    def _write_matrix(m: np.ndarray):
        rows, cols = m.shape
        mat = {}
        mat['rows'] = rows
        mat['cols'] = cols
        mat['data'] = m.flatten().tolist()
        return mat

    def read_yaml(self, file_name):
        """Read mono camera info from a yaml file (ROS) format"""
        cfg = yaml.safe_load(open(file_name, 'r'))
        self.cfg = cfg
        self.image_width = cfg['image_width']
        self.image_height = cfg['image_height']
        self.camera_name = cfg['camera_name']
        self.distortion_model = cfg['distortion_model']
        self.K = self._load_matrix(cfg, 'camera_matrix')
        self.D = self._load_matrix(cfg, 'distortion_coefficients')
        self.R = self._load_matrix(cfg, 'rectification_matrix')
        self.P = self._load_matrix(cfg, 'projection_matrix')

    def write_yaml(self, file_name):
        cfg = {}
        cfg['image_width'] = self.image_width
        cfg['image_height'] = self.image_height
        cfg['camera_name'] = self.camera_name
        cfg['camera_matrix'] = self._write_matrix(self.K)
        cfg['distortion_model'] = 'plumb_bob'
        cfg['distortion_coefficients'] = self._write_matrix(self.D)
        cfg['rectification_matrix'] = self._write_matrix(self.R)
        cfg['projection_matrix'] = self._write_matrix(self.P)
        with open(file_name, 'w') as fid:
            yaml.safe_dump(cfg, fid, sort_keys=False)


class CameraInfoStereo(object):
    def __init__(self):
        print('Not implemented yet')

    def read_yaml(self):
        pass

    def write_yaml(self):
        pass


class CalibratorMono(object):
    def __init__(self, checkerboard:CheckerBoardInfo):
        print('Creating Calibrator Mono')
        self.cam_info = None
        self.checkerboard = checkerboard

    def __del__(self):
        print('Cleaning up calibrator mono')

    def calibrate(self, image_files, show=False):
        # check image sizes
        checkerboard_size = self.checkerboard.size
        objs = self.checkerboard.make_object()

        if not image_files:
            raise ValueError('Image files list is empty!')

        # find corners
        img = cv2.imread(image_files[0])
        img_height, img_width = img.shape[:2]
        print('image width = {} height = {}'.format(img_width, img_height))
        points_image = []
        points_board = []
        images_found = []
        for i, file in enumerate(image_files):
            img = cv2.imread(file)
            if img.shape[:2] != (img_height, img_width):
                print('Image size does not match, please check your data')
                continue
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(img, checkerboard_size, flags=flags)
            print('{:03} {}, found: {}'.format(i, file, found))

            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria)
                points_image.append(corners)
                points_board.append(objs)
                images_found.append(file)

            if show:
                img_show = img
                show_scale = 1.0
                cv2.drawChessboardCorners(img_show, checkerboard_size, corners, found)
                cv2.imshow(file, cv2.resize(img_show, None, fx=show_scale, fy=show_scale))
                cv2.waitKey(500)
                cv2.destroyAllWindows()

        # calibrate
        K = np.eye(3)           # intrinsic camera matrix
        D = np.zeros((1,5))     # distortion matrix (plumb bob) model
        ret, K, D, Rs, Ts = cv2.calibrateCamera(points_board, points_image, (img_width, img_height), K, D,
                                                flags=cv2.CALIB_ZERO_TANGENT_DIST)

        # save cam_info
        cam_info = CameraInfoMono()
        cam_info.image_width = img_width
        cam_info.image_height = img_height
        cam_info.K = K
        cam_info.D = D
        cam_info.Rs = Rs
        cam_info.Ts = Ts
        cam_info.image_files = images_found
        return cam_info

    def verify(self, info):
        reproject_errs = []
        for i, _ in enumerate(info.img_files):
            pts_reproj, _ = cv2.projectPoints(objs, info.Rs[i], info.Ts[i], info.K, info.D)
            err = np.linalg.norm(pts_mono[i] - pts_reproj, axis=(1,2)).mean()
            reproject_errs.append(err)
        reproject_errs = np.array(reproject_errs)
        mean_err = reproject_errs.mean()

        x_pos = np.arange(len(info.img_files))
        x_ticks = ['{:02}'.format(i) for i,_ in enumerate(info.img_files)]
        plt.bar(x_pos, reproject_errs)
        plt.axhline(y=mean_err, color='r')
        plt.xticks(x_pos, x_ticks)
        plt.ylabel('Reproject Error (Pixel)')
        plt.title('Momo camera reprojection errors')
        plt.grid(True, axis='y')
        plt.show()
        print('verifing camera calibration')


class WidgetVerify(QWidget):
    def __init__(self):
        super(WidgetVerify, self).__init__()
        self.init_ui()

    def init_ui(self):
        qpb_verify = QPushButton("Start Verify")
        qpb_verify.clicked.connect(self.slot_verify)
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel())
        vbox.addWidget(qpb_verify)
        self.setLayout(vbox)

    def slot_verify(self):
        pass


class CalibrationWindow(QMainWindow):
    def __init__(self):
        super(CalibrationWindow, self).__init__()
        self.cam = cv2.VideoCapture(0)
        self.calibrator = CalibratorMono()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.slot_timer)
        self.timer.start(50)  # in ms

        self.show()

    def init_ui(self):
        widget = QWidget()
        vbox = QVBoxLayout()
        btn_save = QPushButton("Save Image")
        btn_calib = QPushButton("Calibrate")
        btn_save.pressed.connect(self.slot_save_button_pressed)
        btn_calib.pressed.connect(self.slot_calibrate_button_pressed)

        vbox.addWidget(btn_save)
        vbox.addWidget(btn_calib)
        vbox.addStretch()

        hbox_top = QHBoxLayout()
        self.img_window = QLabel()
        hbox_top.addWidget(self.img_window)
        hbox_top.addLayout(vbox)

        # menu
        action_new = QAction()
        action_new.setShortcut(QKeySequence.New)

        action_about = QAction()
        action_help = QAction()
        action_help.setShortcut(QKeySequence.HelpContents)
        action_help.triggered.connect(self.slot_help)

        menu_bar = self.menuBar()
        menu_help = menu_bar.addMenu("&Help")
        menu_help.addAction(action_about)
        menu_help.addAction(action_help)
        menu_help.addAction(action_new)
        widget.setLayout(hbox_top)

        tab_calibrate = QWidget()
        tab_verify = WidgetVerify()

        tabs = QTabWidget()
        tabs.addTab(widget, "Capture")
        tabs.addTab(tab_calibrate, "Calibrate")
        tabs.addTab(tab_verify, "Verify")

        self.setCentralWidget(tabs)

    def slot_help(self):
        print("Showing help window")

    def slot_timer(self):
        _, frame = self.cam.read()
        h, w, c = frame.shape
        qimg = QImage(frame.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self.img_window.setPixmap(QPixmap(qimg))

    def slot_save_button_pressed(self):
        print("Save button is pressed")
        print("Saving an image")
        # self.calibrator.add_image()

    def slot_calibrate_button_pressed(self):
        print("Calibrate button is pressed")
        print("calibrating ....")

        files = sorted(glob.glob("*.png"))
        # find checker board and display
        # show result
        print("finished calibration!")
        self.calibrator.save_calibration('tmp.yaml')



if __name__ == '__main__':
    checker_board = CheckerBoardInfo(6, 9, 0.02423)
    mono_calibrator = CalibratorMono(checkerboard=checker_board)

    files = sorted(glob.glob('./data/stereo_calibration/left*.jpg'))
    info_left = mono_calibrator.calibrate(files, show=False)
    info_left.write_yaml('./data/stereo_calibration/left.yaml')

    files = sorted(glob.glob('./data/stereo_calibration/right*.jpg'))
    info_right = mono_calibrator.calibrate(files, show=True)
    info_right.write_yaml('./data/stereo_calibration/right.yaml')



# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     win = CalibrationWindow()
#     sys.exit(app.exec_())
