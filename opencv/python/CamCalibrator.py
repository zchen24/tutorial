#!/usr/bin/env python

"""
Camera Calibration Application

Author: Zihan Chen
Date: 2019-08-11
License: LGPL
"""

import os
import sys
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from qtpy.QtWidgets import *
from qtpy.QtGui import *
from qtpy.QtCore import *


# Todo list:
#  - add shortcut 's' to save image
#  - add GUI for mono calibration
#  - mimic the Matlab's GUI

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
        self.Rs = []
        self.Ts = []
        self.cfg = None  # yaml cdf dictionary
        self.image_files = None
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
    def __init__(self, checkerboard:CheckerBoardInfo=None):
        print('Creating Calibrator Mono')
        self.cam_info = None
        self.checkerboard = checkerboard
        self._img_size = None
        self._points_image = []
        self._points_board = []
        self._images_found = []
        self._images_found_filename = []
        self._images_show_chessboard = []

    def __del__(self):
        print('Cleaning up calibrator mono')

    def reset(self):
        self.cam_info = None
        self.checkerboard = None
        self._img_size = None
        self._points_image = []
        self._points_board = []
        self._images_found = []
        self._images_found_filename = []
        self._images_show_chessboard = []

    def detect_chessboard_one(self, image_file, show=False):
        img = cv2.imread(image_file)
        if img is None:
            print('Failed to read image file: {}'.format(image_file))
            return False

        if self._img_size is None:
            self._img_size = img.shape[:2]

        checkerboard_size = self.checkerboard.size
        objs = self.checkerboard.make_object()
        if img.shape[:2] != self._img_size:
            print('Image size does not match, please check your data')
            return False

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(img, checkerboard_size, flags=flags)
        print('{}, found: {}'.format(image_file, found))

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria)
            self._points_image.append(corners)
            self._points_board.append(objs)
            self._images_found.append(img)
            self._images_found_filename.append(image_file)
            img_show = img
            cv2.drawChessboardCorners(img_show, checkerboard_size, corners, found)
            self._images_show_chessboard.append(img_show)

        if show:
            img_show = img
            show_scale = 1.0
            cv2.imshow(image_file, cv2.resize(img_show, None, fx=show_scale, fy=show_scale))
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    def detect_chessboard(self, image_files, show=False):
        # check image sizes
        if not image_files:
            raise ValueError('Image files list is empty!')
        # find corners
        img = cv2.imread(image_files[0])
        self._img_height, self._img_width = img.shape[:2]
        print('image width = {} height = {}'.format(self._img_width, self._img_height))
        self._points_image = []
        self._points_board = []
        self._images_found_filename = []
        for i, file in enumerate(image_files):
            self.detect_chessboard_one(file, show)

    def calibrate(self, img_size=None, points_board=None, points_image=None, flags=None):
        if img_size is None:
            img_height, img_width = self._img_size
        else:
            img_height, img_width,  = img_size

        if points_board is None:
            points_board = self._points_board

        if points_image is None:
            points_image = self._points_image

        if flags is None:
            flags = cv2.CALIB_ZERO_TANGENT_DIST

        # calibrate
        K = np.eye(3)           # intrinsic camera matrix
        D = np.zeros((1,5))     # distortion matrix (plumb bob) model
        ret, K, D, Rs, Ts = cv2.calibrateCamera(points_board, points_image, (img_width, img_height), K, D,
                                                flags=flags)

        # save cam_info
        cam_info = CameraInfoMono()
        cam_info.image_width = img_width
        cam_info.image_height = img_height
        cam_info.K = K
        cam_info.D = D
        cam_info.Rs = Rs
        cam_info.Ts = Ts
        cam_info.image_files = self._images_found_filename
        self.cam_info = cam_info
        return cam_info

    def verify(self, info: CameraInfoMono):
        print('verifying camera calibration')
        objs = self.checkerboard.make_object()
        reproject_errs = []
        for i, _ in enumerate(info.image_files):
            pts_mono = self._points_image[i]
            pts_reproj, _ = cv2.projectPoints(objs, info.Rs[i], info.Ts[i], info.K, info.D)
            err = np.linalg.norm(pts_mono - pts_reproj, axis=(1,2)).mean()
            reproject_errs.append(err)
        reproject_errs = np.array(reproject_errs)
        mean_err = reproject_errs.mean()

        x_pos = np.arange(len(info.image_files))
        x_ticks = ['{:02}'.format(i) for i,_ in enumerate(info.image_files)]
        plt.bar(x_pos, reproject_errs)
        plt.axhline(y=mean_err, color='r')
        plt.xticks(x_pos, x_ticks)
        plt.ylabel('Reproject Error (Pixel)')
        plt.title('Mono camera reprojection errors')
        plt.grid(True, axis='y')
        plt.show()



class CalibratorStereo(object):
    def __init__(self):
        print('Calibrator Stereo')


class DialogCheckerBoardInfo(QDialog):
    def __init__(self):
        super(DialogCheckerBoardInfo, self).__init__()

        grid = QGridLayout()
        self.edit_width = QLineEdit()
        self.edit_width.setValidator(QIntValidator(0, 20))
        self.edit_width.setText(str(9))
        self.edit_height = QLineEdit()
        self.edit_height.setValidator(QIntValidator(0, 20))
        self.edit_height.setText(str(6))
        self.edit_dim = QLineEdit()
        self.edit_dim.setValidator(QDoubleValidator(0.0, 1000.0, int(2)))
        self.edit_dim.setText(str(5))

        grid.addWidget(QLabel('Width'), 0, 0)
        grid.addWidget(self.edit_width, 0, 1)
        grid.addWidget(QLabel('Height'), 1, 0)
        grid.addWidget(self.edit_height, 1, 1)
        grid.addWidget(QLabel('Size (mm)'), 2, 0)
        grid.addWidget(self.edit_dim, 2, 1)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        btn_ok = QPushButton('OK')
        btn_ok.clicked.connect(self.accept)
        hbox.addWidget(btn_ok)
        hbox.addWidget(QPushButton('Cancel'))
        vbox.addLayout(grid)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.setWindowTitle('Checker Board Info')
        self.setWindowModality(Qt.WindowModal)

    def get_checkerboard_info(self):
        return CheckerBoardInfo(int(self.edit_width.text()),
                                int(self.edit_height.text()),
                                float(self.edit_dim.text()) * 0.001)


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


class TabCalibrate(QWidget):
    def __init__(self):
        super(TabCalibrate, self).__init__()
        self.calibrator = None

        self.img_list = QListWidget()
        self.img_list.setMinimumWidth(100)
        self.img_list.currentRowChanged.connect(self.slot_data_selected_changed)

        self.img_selected = QLabel()
        self.img_selected.setMinimumSize(400, 400)

        btn_add_img = QPushButton("Add Images")
        btn_add_img.pressed.connect(self.slot_addimage_button_pressed)
        btn_calibrate = QPushButton("Calibrate")
        btn_calibrate.pressed.connect(self.slot_calibrate_pressed)
        self.btn_save = QPushButton("Save Camera Parameters")
        self.btn_save.pressed.connect(self.slot_save_cam_param_pressed)
        self.btn_save.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(btn_add_img)
        vbox.addWidget(btn_calibrate)
        vbox.addWidget(self.btn_save)
        vbox.addStretch()

        hbox = QHBoxLayout()
        hsplitter = QSplitter(Qt.Horizontal)
        hsplitter.addWidget(self.img_list)
        hsplitter.addWidget(self.img_selected)
        hbox.addWidget(hsplitter)
        hbox.addLayout(vbox)
        self.setLayout(hbox)

    def _update_image_browser(self):
        self.img_list.setIconSize(QSize(72, 72))
        self.img_list.setAcceptDrops(True)

        for i, img in enumerate(self.calibrator._images_found):
            # self.img_list.iconSize()
            thumbnail_size = self.img_list.iconSize()
            img_thumbnail = cv2.resize(img, (thumbnail_size.width(), thumbnail_size.height()))
            h, w, c = img_thumbnail.shape
            q_img = QImage(img_thumbnail.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
            icon = QIcon(QPixmap(q_img))
            filename = os.path.basename(self.calibrator._images_found_filename[i])
            self.img_list.addItem(QListWidgetItem(icon, '{}\n{}'.format(i, filename)))

        self.img_list.setCurrentRow(0)

    def slot_data_selected_changed(self, currentRow:int):
        print('Current Row = {}'.format(currentRow))
        img = self.calibrator._images_found[currentRow]
        h, w, c = img.shape
        q_img = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        self.img_selected.setPixmap(QPixmap(q_img))

    def slot_addimage_button_pressed(self):
        print("Add Images button is pressed")
        self._pause_preview = True
        # select files / folder
        options = QFileDialog.Options()
        caption = "Select image files"
        directory = "./"
        filters = "All Files (*);;Images (*.png *.jpg)"
        initial_filter = "Images (*.png *.jpg)"
        filenames, _ = QFileDialog.getOpenFileNames(None,
                                                    caption,
                                                    directory,
                                                    filters,
                                                    initial_filter,
                                                    options)

        print('Selected files: {}'.format(filenames))

        # specify checker board info
        dialog_info = DialogCheckerBoardInfo()
        if dialog_info.exec_():
            checkerboard_info = dialog_info.get_checkerboard_info()
            print(checkerboard_info)
            self.calibrator = CalibratorMono(checkerboard_info)

        # find checkerboard
        num_files = len(filenames)
        progress = QProgressDialog('Detecting Checkerboard', 'Abort', 0, num_files, parent=None)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        for i in range(num_files):
            progress.setValue(i)
            self.calibrator.detect_chessboard_one(filenames[i], True)
            if progress.wasCanceled():
                print('Copying files cancelled')
                break
        # show result
        self._update_image_browser()
        self._pause_preview = False
        print("Added images for calibration!")

    def slot_calibrate_pressed(self):
        print('Calibrating')
        if self.calibrator is not None:
            cam_info = self.calibrator.calibrate()
            self.btn_save.setEnabled(True)
        print('Calibrating DONE')

    def slot_save_cam_param_pressed(self):
        if self.calibrator is not None and self.calibrator.cam_info is not None:
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.AnyFile)
            options = QFileDialog.Options()
            caption = "Save configuration file to "
            directory = "./"
            filters = "All Files (*);;Configuration (*.yaml *.yml)"
            initial_filter = "Configuration (*.yaml *.yml)"
            filename, _ = dialog.getSaveFileName(self,
                                                 caption,
                                                 directory,
                                                 filters,
                                                 initial_filter,
                                                 options)
            self.calibrator.cam_info.write_yaml(filename)
            print('Save camera parameters to {}'.format(filename))

class CalibrationWindow(QMainWindow):
    def __init__(self):
        super(CalibrationWindow, self).__init__()
        self.cam = cv2.VideoCapture(0)
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.slot_timer)
        self.timer.start(50)  # in ms

        self.num_image_saved = 0
        self._pause_preview = False
        self.show()

    def init_ui(self):
        widget = QWidget()
        vbox = QVBoxLayout()
        btn_save = QPushButton("Save Image")
        btn_save.pressed.connect(self.slot_save_button_pressed)

        vbox.addWidget(btn_save)
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

        tab_calibrate = TabCalibrate()
        tabs = QTabWidget()
        tabs.addTab(widget, "Capture")
        tabs.addTab(tab_calibrate, "Calibrate")

        self.setCentralWidget(tabs)

    def slot_help(self):
        print("Showing help window")

    def slot_timer(self):
        if not self._pause_preview:
            ret, frame = self.cam.read()
            if ret:
                h, w, c = frame.shape
                q_img = QImage(frame.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
                self.img_window.setPixmap(QPixmap(q_img))

    def slot_save_button_pressed(self):
        ret, img = self.cam.read()
        if ret:
            filename = 'image_{:02}.png'.format(self.num_image_saved)
            self.num_image_saved += 1
            cv2.imwrite(filename, img)
            print("Saving an image: {}".format(filename))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CalibrationWindow()
    sys.exit(app.exec_())


# if __name__ == '__main__':
#     import glob
#     checker_board = CheckerBoardInfo(6, 9, 0.02423)
#     mono_calibrator = CalibratorMono(checkerboard=checker_board)
#
#     files = sorted(glob.glob('./data/stereo_calibration/left*.jpg'))
#     mono_calibrator.detect_chessboard(files, show=True)
#     info_left = mono_calibrator.calibrate()
#     info_left.write_yaml('./data/stereo_calibration/left.yaml')
#     import ipdb; ipdb.set_trace()
#     mono_calibrator.verify(info_left)
