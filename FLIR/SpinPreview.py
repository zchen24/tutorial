#!/usr/bin/env python3

"""
Preview Video Stream using OpenCV
"""

import PySpin
import cv2


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

print('*** IMAGE ACQUISITION ***\n')
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
print('Acquisition mode set to continuous...')

cam.BeginAcquisition()
cam_serial_number = cam.DeviceSerialNumber.ToString()
print('Camera {} start acquisition'.format(cam_serial_number))

while True:
    img = cam.GetNextImage()
    if img.IsIncomplete():
        print('Image incomplete with image status %d ...' % img.GetImageStatus())
        continue

    width = img.GetWidth()
    height = img.GetHeight()
    img_converted = img.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
    img_cv = img_converted.GetData().reshape((height, width, 3))
    img.Release()
    cv2.imshow('Preview', img_cv)
    if cv2.waitKey(5) == ord('q'):
        cv2.destroyAllWindows()
        break

cam.EndAcquisition()

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
