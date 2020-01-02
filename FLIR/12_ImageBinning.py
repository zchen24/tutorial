#!/usr/bin/env python3

"""
Shows how to use image binning to reduce image size
Date: 2020-01-01
"""

import PySpin


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

print('Image default size: {} x {}'.format(cam.Width.GetMax(),
                                           cam.Height.GetMax()))

cam.BinningSelector.SetValue(PySpin.BinningSelector_All)
cam.BinningHorizontalMode.SetValue(PySpin.BinningHorizontalMode_Average)
cam.BinningVerticalMode.SetValue(PySpin.BinningHorizontalMode_Average)
cam.BinningHorizontal.SetValue(2)
cam.BinningVertical.SetValue(2)

print('After binning setup:')
print('Image size: {} x {}'.format(cam.Width.GetValue(),
                                   cam.Height.GetValue()))

print('*** IMAGE ACQUISITION ***\n')
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
print('Acquisition mode set to continuous...')

cam.BeginAcquisition()
cam_serial_number = cam.DeviceSerialNumber.ToString()
img = cam.GetNextImage()
filename = 'ImageBinning.jpg'
img.Save(filename)
print('Image size {} x {}, saving to {}'.format(img.GetWidth(),
                                                img.GetHeight(),
                                                filename))
img.Release()
cam.EndAcquisition()

# reset binning
cam.BinningHorizontal.SetValue(1)
cam.BinningVertical.SetValue(1)

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
