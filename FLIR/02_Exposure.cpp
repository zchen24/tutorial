#!/usr/bin/env python3

"""
Shows how to manually change camera's exposure time.
See Exposure_QuickSpin.py example

This program sets camera to different exposure time, then take an
image. Note the buffer handling mode is set to NewestOnly to make
sure the image retrieved is taken after the exposure time has been
changed.

Author: Zihan Chen
Date: 2019-12-29
"""

import time
import PySpin


def set_exposure(cam: PySpin.CameraPtr, exposure_ms: int):
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    exp_max = cam.ExposureTime.GetMax()
    if exposure_ms > exp_max:
        exposure_ms = exp_max
        print('Exposure set value exceeds limit, setting to max {}'.format(exp_max))
    elif exposure_ms <= 0:
        exposure_ms = 1
        print('Exposure time must be a positive int value')
    cam.ExposureTime.SetValue(exposure_ms)
    print('Set exposure to {} ms'.format(exposure_ms))


def reset_exposure(cam: PySpin.CameraPtr):
    try:
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
        print('Automatic exposure enabled...')
        return True
    except PySpin.SpinnakerException as e:
        print('Error: {}'.format(e))
        return False


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

print('*** IMAGE ACQUISITION ***\n')
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
print('Acquisition mode set to continuous...')

cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestOnly)
print('StreamBuffer handling mode set to NewestOnly...')

cam.BeginAcquisition()
cam_serial_number = cam.DeviceSerialNumber.ToString()

exposure_all = [1000, 2000, 3000, 4000, 5000]

for exposure in exposure_all:
    set_exposure(cam, exposure)
    time.sleep(0.05)
    img = cam.GetNextImage()
    if not img.IsIncomplete():
        img.Save('Exposure_{}.jpg'.format(exposure))
    img.Release()

cam.EndAcquisition()
reset_exposure(cam)

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
