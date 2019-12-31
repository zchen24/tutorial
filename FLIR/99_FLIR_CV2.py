#!/usr/bin/env python3

"""
Shows how to convert PySpin image to OpenCV image
Date: 2019-12-31
"""

import PySpin
import cv2

system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
cam.BeginAcquisition()
cam_serial_number = cam.DeviceSerialNumber.ToString()

img = cam.GetNextImage()
width = img.GetWidth()
height = img.GetHeight()
img_cv = img.GetData().reshape((height, width))
cv2.imwrite('FLIR_CV2.jpg', img_cv)
img.Release()

cam.EndAcquisition()

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
